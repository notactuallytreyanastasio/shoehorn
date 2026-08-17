//! Minimal GGUF v3 reader/writer.
//!
//! Reads the header, metadata KVs, and tensor infos; tensor data is accessed
//! through the caller's mmap using the computed data-section offset. The writer
//! emits a fresh file preserving arbitrary KVs.

use anyhow::{bail, Context, Result};
use std::io::{BufWriter, Write};

pub const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
pub const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlType {
    F32,
    F16,
    Bf16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
    Iq2Xxs,
    Iq2Xs,
    Iq2S,
    Iq3Xxs,
    Iq3S,
    Iq4Nl,
    Iq4Xs,
    Other(u32),
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            16 => Self::Iq2Xxs,
            17 => Self::Iq2Xs,
            18 => Self::Iq3Xxs,
            20 => Self::Iq4Nl,
            21 => Self::Iq3S,
            22 => Self::Iq2S,
            23 => Self::Iq4Xs,
            30 => Self::Bf16,
            other => Self::Other(other),
        }
    }

    pub fn to_u32(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Iq2Xxs => 16,
            Self::Iq2Xs => 17,
            Self::Iq3Xxs => 18,
            Self::Iq4Nl => 20,
            Self::Iq3S => 21,
            Self::Iq2S => 22,
            Self::Iq4Xs => 23,
            Self::Bf16 => 30,
            Self::Other(v) => v,
        }
    }

    /// (elements per block, bytes per block)
    pub fn block_layout(self) -> (u64, u64) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 | Self::Bf16 => (1, 2),
            Self::Q4_0 => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::Q5_0 => (32, 22),
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),
            Self::Q4K => (256, 144),
            Self::Q5K => (256, 176),
            Self::Q6K => (256, 210),
            Self::Iq2Xxs => (256, 66),
            Self::Iq2Xs => (256, 74),
            Self::Iq2S => (256, 82),
            Self::Iq3Xxs => (256, 98),
            Self::Iq3S => (256, 110),
            Self::Iq4Nl => (32, 18),
            Self::Iq4Xs => (256, 136),
            Self::Other(_) => (1, 0),
        }
    }

    pub fn row_bytes(self, n_per_row: u64) -> u64 {
        let (bs, tb) = self.block_layout();
        assert!(n_per_row.is_multiple_of(bs), "row size {n_per_row} not divisible by block {bs}");
        n_per_row / bs * tb
    }

    pub fn name(self) -> String {
        match self {
            Self::F32 => "F32".into(),
            Self::F16 => "F16".into(),
            Self::Bf16 => "BF16".into(),
            Self::Q4_0 => "Q4_0".into(),
            Self::Q4_1 => "Q4_1".into(),
            Self::Q5_0 => "Q5_0".into(),
            Self::Q5_1 => "Q5_1".into(),
            Self::Q8_0 => "Q8_0".into(),
            Self::Q4K => "Q4_K".into(),
            Self::Q5K => "Q5_K".into(),
            Self::Q6K => "Q6_K".into(),
            Self::Iq2Xxs => "IQ2_XXS".into(),
            Self::Iq2Xs => "IQ2_XS".into(),
            Self::Iq2S => "IQ2_S".into(),
            Self::Iq3Xxs => "IQ3_XXS".into(),
            Self::Iq3S => "IQ3_S".into(),
            Self::Iq4Nl => "IQ4_NL".into(),
            Self::Iq4Xs => "IQ4_XS".into(),
            Self::Other(v) => format!("type{v}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    /// (element type id, values)
    Arr(u32, Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v as u64),
            Value::U16(v) => Some(v as u64),
            Value::U32(v) => Some(v as u64),
            Value::U64(v) => Some(v),
            Value::I8(v) if v >= 0 => Some(v as u64),
            Value::I16(v) if v >= 0 => Some(v as u64),
            Value::I32(v) if v >= 0 => Some(v as u64),
            Value::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    fn type_id(&self) -> u32 {
        match self {
            Value::U8(_) => 0,
            Value::I8(_) => 1,
            Value::U16(_) => 2,
            Value::I16(_) => 3,
            Value::U32(_) => 4,
            Value::I32(_) => 5,
            Value::F32(_) => 6,
            Value::Bool(_) => 7,
            Value::Str(_) => 8,
            Value::Arr(..) => 9,
            Value::U64(_) => 10,
            Value::I64(_) => 11,
            Value::F64(_) => 12,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// Logical dims, ne[0] is the row (contiguous) dimension.
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Offset relative to the start of the data section.
    pub offset: u64,
    /// Which shard of a split model holds this tensor's data.
    pub shard: usize,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }
    pub fn ne0(&self) -> u64 {
        self.dims[0]
    }
    pub fn n_rows(&self) -> u64 {
        self.dims.iter().skip(1).product()
    }
    pub fn byte_size(&self) -> u64 {
        self.ty.row_bytes(self.ne0()) * self.n_rows()
    }
}

pub struct GgufFile {
    pub kvs: Vec<(String, Value)>,
    pub tensors: Vec<TensorInfo>,
    /// Absolute file offset where the data section begins.
    pub data_start: u64,
    pub alignment: u64,
}

impl GgufFile {
    pub fn tensor_data<'a>(&self, buf: &'a [u8], t: &TensorInfo) -> &'a [u8] {
        let start = (self.data_start + t.offset) as usize;
        &buf[start..start + t.byte_size() as usize]
    }
}

/// A model on disk: one GGUF file, or a llama.cpp-style split
/// (`-00001-of-0000N.gguf`) opened as a single logical model. Owns the mmaps;
/// tensor offsets are rewritten to be absolute within their shard.
pub struct Model {
    shards: Vec<memmap2::Mmap>,
    pub kvs: Vec<(String, Value)>,
    pub tensors: Vec<TensorInfo>,
    pub alignment: u64,
}

impl Model {
    pub fn open(path: &std::path::Path) -> Result<Model> {
        let paths = split_siblings(path)?;
        let mut shards = Vec::new();
        let mut kvs = Vec::new();
        let mut tensors = Vec::new();
        let mut alignment = DEFAULT_ALIGNMENT;
        for (si, p) in paths.iter().enumerate() {
            let f = std::fs::File::open(p).with_context(|| format!("opening {}", p.display()))?;
            let map = unsafe { memmap2::Mmap::map(&f)? };
            let g = read(&map).with_context(|| format!("parsing {}", p.display()))?;
            if si == 0 {
                kvs = g.kvs;
                alignment = g.alignment;
            }
            for mut t in g.tensors {
                t.offset += g.data_start; // absolute within this shard
                t.shard = si;
                tensors.push(t);
            }
            shards.push(map);
        }
        if paths.len() > 1 {
            eprintln!("opened split model: {} shards, {} tensors", paths.len(), tensors.len());
        }
        Ok(Model { shards, kvs, tensors, alignment })
    }

    pub fn kv(&self, key: &str) -> Option<&Value> {
        self.kvs.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn tensor_data(&self, t: &TensorInfo) -> &[u8] {
        let start = t.offset as usize;
        &self.shards[t.shard][start..start + t.byte_size() as usize]
    }

    pub fn total_bytes(&self) -> u64 {
        self.shards.iter().map(|s| s.len() as u64).sum()
    }
}

/// For `x-00001-of-0000N.gguf`, return all N shard paths; otherwise just `path`.
fn split_siblings(path: &std::path::Path) -> Result<Vec<std::path::PathBuf>> {
    let name = path.file_name().map(|s| s.to_string_lossy()).unwrap_or_default().into_owned();
    let Some(caps) = name
        .strip_suffix(".gguf")
        .and_then(|s| s.rsplit_once("-of-"))
        .and_then(|(head, count)| {
            let (stem, no) = head.rsplit_once('-')?;
            let no: usize = no.parse().ok().filter(|_| no.len() == 5)?;
            let count: usize = count.parse().ok().filter(|_| count.len() == 5)?;
            Some((stem.to_string(), no, count))
        })
    else {
        return Ok(vec![path.to_path_buf()]);
    };
    let (stem, no, count) = caps;
    if count <= 1 {
        return Ok(vec![path.to_path_buf()]);
    }
    if no != 1 {
        bail!("open the first shard of a split model (…-00001-of-{count:05}.gguf)");
    }
    let dir = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let mut out = Vec::with_capacity(count);
    for i in 1..=count {
        let p = dir.join(format!("{stem}-{i:05}-of-{count:05}.gguf"));
        if !p.exists() {
            bail!("missing shard {}", p.display());
        }
        out.push(p);
    }
    Ok(out)
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            bail!("unexpected EOF at {} (+{n})", self.pos);
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(len)?).into_owned())
    }
    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.take(1)?[0]),
            1 => Value::I8(self.take(1)?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.take(2)?.try_into()?)),
            3 => Value::I16(i16::from_le_bytes(self.take(2)?.try_into()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_le_bytes(self.take(4)?.try_into()?)),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let elem_ty = self.u32()?;
                let count = self.u64()? as usize;
                let mut vals = Vec::with_capacity(count.min(1 << 24));
                for _ in 0..count {
                    vals.push(self.value(elem_ty)?);
                }
                Value::Arr(elem_ty, vals)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_le_bytes(self.take(8)?.try_into()?)),
            other => bail!("unknown GGUF value type {other}"),
        })
    }
}

pub fn read(buf: &[u8]) -> Result<GgufFile> {
    let mut r = Reader { buf, pos: 0 };
    let magic = r.u32().context("reading magic")?;
    if magic != GGUF_MAGIC {
        bail!("not a GGUF file (magic {magic:#x})");
    }
    let version = r.u32()?;
    if !(2..=3).contains(&version) {
        bail!("unsupported GGUF version {version}");
    }
    let n_tensors = r.u64()? as usize;
    let n_kv = r.u64()? as usize;

    let mut kvs = Vec::with_capacity(n_kv);
    for _ in 0..n_kv {
        let key = r.string()?;
        let ty = r.u32()?;
        let val = r.value(ty).with_context(|| format!("reading value for key {key}"))?;
        kvs.push((key, val));
    }

    let mut tensors = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        let name = r.string()?;
        let n_dims = r.u32()? as usize;
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(r.u64()?);
        }
        let ty = GgmlType::from_u32(r.u32()?);
        let offset = r.u64()?;
        tensors.push(TensorInfo { name, dims, ty, offset, shard: 0 });
    }

    let alignment = kvs
        .iter()
        .find(|(k, _)| k == "general.alignment")
        .and_then(|(_, v)| v.as_u64())
        .unwrap_or(DEFAULT_ALIGNMENT);
    let data_start = (r.pos as u64).div_ceil(alignment) * alignment;

    Ok(GgufFile { kvs, tensors, data_start, alignment })
}

fn write_string<W: Write>(w: &mut W, s: &str) -> Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())?;
    Ok(())
}

fn write_value<W: Write>(w: &mut W, v: &Value) -> Result<()> {
    match v {
        Value::U8(x) => w.write_all(&[*x])?,
        Value::I8(x) => w.write_all(&[*x as u8])?,
        Value::U16(x) => w.write_all(&x.to_le_bytes())?,
        Value::I16(x) => w.write_all(&x.to_le_bytes())?,
        Value::U32(x) => w.write_all(&x.to_le_bytes())?,
        Value::I32(x) => w.write_all(&x.to_le_bytes())?,
        Value::F32(x) => w.write_all(&x.to_le_bytes())?,
        Value::Bool(x) => w.write_all(&[*x as u8])?,
        Value::Str(s) => write_string(w, s)?,
        Value::U64(x) => w.write_all(&x.to_le_bytes())?,
        Value::I64(x) => w.write_all(&x.to_le_bytes())?,
        Value::F64(x) => w.write_all(&x.to_le_bytes())?,
        Value::Arr(elem_ty, vals) => {
            w.write_all(&elem_ty.to_le_bytes())?;
            w.write_all(&(vals.len() as u64).to_le_bytes())?;
            for v in vals {
                write_value(w, v)?;
            }
        }
    }
    Ok(())
}

/// Write a GGUF v3 file with bounded memory: tensor sizes are known from the
/// infos, so the header goes out first and `produce(i)` supplies each tensor's
/// data in order, to be written and dropped immediately.
pub fn write_streaming<W: Write, F: FnMut(usize) -> Result<Vec<u8>>>(
    out: W,
    kvs: &[(String, Value)],
    infos: &[TensorInfo],
    alignment: u64,
    mut produce: F,
) -> Result<u64> {
    let mut w = BufWriter::with_capacity(1 << 20, out);
    w.write_all(&GGUF_MAGIC.to_le_bytes())?;
    w.write_all(&3u32.to_le_bytes())?;
    w.write_all(&(infos.len() as u64).to_le_bytes())?;
    w.write_all(&(kvs.len() as u64).to_le_bytes())?;

    for (k, v) in kvs {
        write_string(&mut w, k)?;
        w.write_all(&v.type_id().to_le_bytes())?;
        write_value(&mut w, v)?;
    }

    // Assign aligned offsets in order from the known byte sizes.
    let mut offset = 0u64;
    let mut offsets = Vec::with_capacity(infos.len());
    for info in infos {
        offset = offset.div_ceil(alignment) * alignment;
        offsets.push(offset);
        offset += info.byte_size();
    }

    for (info, off) in infos.iter().zip(&offsets) {
        write_string(&mut w, &info.name)?;
        w.write_all(&(info.dims.len() as u32).to_le_bytes())?;
        for d in &info.dims {
            w.write_all(&d.to_le_bytes())?;
        }
        w.write_all(&info.ty.to_u32().to_le_bytes())?;
        w.write_all(&off.to_le_bytes())?;
    }

    // Pad to data-section alignment. BufWriter doesn't track position, so
    // recompute the header size analytically via a counting pass.
    let mut header_len = 4 + 4 + 8 + 8;
    {
        let mut counter = CountWriter(0);
        for (k, v) in kvs {
            write_string(&mut counter, k)?;
            counter.write_all(&v.type_id().to_le_bytes())?;
            write_value(&mut counter, v)?;
        }
        for info in infos {
            write_string(&mut counter, &info.name)?;
            counter.write_all(&[0u8; 4])?;
            for _ in &info.dims {
                counter.write_all(&[0u8; 8])?;
            }
            counter.write_all(&[0u8; 12])?;
        }
        header_len += counter.0;
    }
    let data_start = header_len.div_ceil(alignment) * alignment;
    w.write_all(&vec![0u8; (data_start - header_len) as usize])?;

    let mut pos = 0u64;
    for (i, (info, off)) in infos.iter().zip(&offsets).enumerate() {
        if *off > pos {
            w.write_all(&vec![0u8; (*off - pos) as usize])?;
            pos = *off;
        }
        let data = produce(i)?;
        assert_eq!(
            data.len() as u64,
            info.byte_size(),
            "tensor {} data size mismatch: {} vs {}",
            info.name,
            data.len(),
            info.byte_size()
        );
        w.write_all(&data)?;
        pos += data.len() as u64;
    }
    w.flush()?;
    Ok(data_start + pos)
}

struct CountWriter(u64);
impl Write for CountWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0 += buf.len() as u64;
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
