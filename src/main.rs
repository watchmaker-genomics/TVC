use clap::Parser;
use clap::ValueEnum;
// REVIEW: is the cfg feature needed all three times?
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use rust_htslib::bam::pileup::Alignment;
use rust_htslib::bam::pileup::Indel;
use rust_htslib::bam::pileup::Pileup;
use rust_htslib::bam::record::Cigar;
use rust_htslib::bam::{self, Read};
use rust_htslib::faidx;
use statrs::distribution::{Binomial, Discrete, DiscreteCDF};
#[cfg(feature = "onnx-inference")]
use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::Path;
use std::sync::OnceLock;
use tracing_subscriber::fmt as subscriber_fmt;
use tracing_subscriber::EnvFilter;
use tracing::info;
#[cfg(feature = "onnx-inference")]
use tracing::warn;
// REVIEW: should some of this be moved
#[cfg(feature = "onnx-inference")]
use ort::{session::Session as OrtSession, value::TensorRef};

#[cfg(feature = "onnx-inference")]

thread_local! {
    static THREAD_LOCAL_ONNX_MODELS: RefCell<HashMap<String, Option<OrtSession>>> = RefCell::new(HashMap::new());
}
#[cfg(feature = "onnx-inference")]
const MODEL_TNC_BASES: [char; 5] = ['A', 'C', 'G', 'T', 'N'];
#[cfg(feature = "onnx-inference")]
const MODEL_VT_VALUES: [&str; 5] = ["COMPLEX", "DEL", "INS", "MNP", "SNP"];

// ---------------------------------------------------------------------------
// CLI and configuration
// ---------------------------------------------------------------------------

/// Scalar (non-one-hot) features the caller can generate for a model.
/// IMPORTANT: names must match those used in `build_model_feature_map`.
#[cfg(feature = "onnx-inference")]
const BASE_FEATURE_NAMES: &[&str] = &[
    "DP", "AO", "ER", "PR",
    "AMQR_R1", "AMQR_R2", "AMQA_R1", "AMQA_R2",
    "ABQR_R1", "ABQR_R2", "ABQA_R1", "ABQA_R2",
    "REDR_R1", "REDR_R2", "REDA_R1", "REDA_R2",
    "ISR", "ISA",
    "FWDP", "REVP", "LLE", "SLE",
    "AMPR_R1", "AMPR_R2", "ARL_R1", "ARL_R2",
    "FWD_R1", "FWD_R2", "REV_R1", "REV_R2", "TOT_R1", "TOT_R2",
    "AF", "strand_bias",
];

#[derive(Debug, Clone, PartialEq, ValueEnum)]
pub enum ReadNumber {
    R1,
    R2,
}

#[derive(Debug, Clone, PartialEq, ValueEnum)]
enum SequencingMode {
    SingleEnd,
    PairedEnd,
}

#[derive(ValueEnum, Clone, Debug)]
enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl LogLevel {
    fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Error => "error",
            LogLevel::Warn => "warn",
            LogLevel::Info => "info",
            LogLevel::Debug => "debug",
            LogLevel::Trace => "trace",
        }
    }
}

/// Input arguments for the Taps Variant Caller
#[derive(Parser, Debug)]
#[command(name = "tvc", version, about = "A Taps+ Variant Caller")]
struct Args {
    input_ref: String,
    input_bam: String,
    output_vcf: String,

    #[arg(long)]
    matched_normal_bam: Option<String>,

    #[arg(short = 'b', long, default_value_t = 20)]
    min_bq: usize,

    #[arg(short = 'm', long, default_value_t = 1)]
    min_mapq: usize,

    #[arg(short = 'd', long, default_value_t = 2)]
    min_depth: u32,

    #[arg(short = 'e', long, default_value_t = 5)]
    end_of_read_cutoff: usize,

    #[arg(short = 'i', long, default_value_t = 0)]
    indel_end_of_read_cutoff: usize,

    #[arg(short = 'x', long, default_value_t = 10)]
    max_mismatches: u32,

    #[arg(short = 'a', long, default_value_t = 2)]
    min_ao: u32,

    #[arg(short = 't', long, default_value_t = 4)]
    num_threads: usize,

    #[arg(short = 'c', long, default_value_t = 1000000)]
    chunk_size: u64,

    #[arg(short = 'p', long, default_value_t = 0.005)]
    error_rate: f64,

    #[arg(short = 'f', long, default_value_t = 3)]
    indel_filter_repeat_limit: usize,

    #[arg(short = 'r', long, value_enum, default_value_t = ReadNumber::R1)]
    stranded_read: ReadNumber,

    #[arg(long = "sequencing-mode", value_enum, default_value_t = SequencingMode::PairedEnd)]
    sequencing_mode: SequencingMode,

    #[arg(short = 'l', long, value_enum, default_value_t = LogLevel::Info)]
    log_level: LogLevel,

    #[arg(short = 'k', long, default_value = "model.onnx")]
    model_path: String,

    #[arg(short = 'n', long = "tumor-ml-threshold", alias = "ml-threshold", default_value_t = 0.99)]
    tumor_ml_threshold: f64,

    #[arg(long, default_value_t = 0.00)]
    normal_ml_threshold: f64,
}

/// Representation of a genomic variant
///
/// # Fields
/// * `contig` - Chromosome or contig name
/// * `pos` - 1-based position of the variant
/// * `reference` - Reference allele
/// * `alt` - Alternate allele
/// * `genotype` - Genotype string (e.g., "0/1")
/// * `score` - Phred-scaled quality score
/// * `depth` - Read depth at the variant position
/// * `alt_counts` - Count of reads supporting the alternate allele
/// * `calling_directive` - Calling directive for the variant caller
///
/// The remaining fields are the per-site features reported in the VCF FORMAT
/// column and fed to the ML model.
#[derive(Clone, Debug)]
struct Variant {
    contig: String,
    pos: u32,
    reference: String,
    alt: String,
    genotype: String,
    score: f64,
    depth: u32,
    alt_counts: u32,
    calling_directive: CallingDirective,
    error_rate: f64,
    tnc: TrinucleotideContext,
    probability: f64,
    average_ref_mapq: f64,
    average_alt_mapq: f64,
    average_ref_bq: f64,
    average_alt_bq: f64,
    average_ref_mapq_r1: f64,
    average_ref_mapq_r2: f64,
    average_alt_mapq_r1: f64,
    average_alt_mapq_r2: f64,
    average_ref_bq_r1: f64,
    average_ref_bq_r2: f64,
    average_alt_bq_r1: f64,
    average_alt_bq_r2: f64,
    avg_ref_dist_from_read_end: f64,
    avg_alt_dist_from_read_end: f64,
    #[cfg_attr(not(feature = "onnx-inference"), allow(dead_code))]
    avg_ref_dist_from_read_end_r1: f64,
    #[cfg_attr(not(feature = "onnx-inference"), allow(dead_code))]
    avg_ref_dist_from_read_end_r2: f64,
    #[cfg_attr(not(feature = "onnx-inference"), allow(dead_code))]
    avg_alt_dist_from_read_end_r1: f64,
    #[cfg_attr(not(feature = "onnx-inference"), allow(dead_code))]
    avg_alt_dist_from_read_end_r2: f64,
    avg_ref_insert_size: f64,
    avg_alt_insert_size: f64,
    fwd_probability: f64,
    rev_probability: f64,
    large_local_entropy: f64,
    small_local_entropy: f64,
    avg_mismatch_per_read: f64,
    avg_mismatch_per_read_r1: f64,
    avg_mismatch_per_read_r2: f64,
    avg_read_length: f64,
    avg_read_length_r1: f64,
    avg_read_length_r2: f64,
    alt_forward_count: u32,
    alt_forward_count_r1: u32,
    alt_forward_count_r2: u32,
    alt_reverse_count: u32,
    alt_reverse_count_r1: u32,
    alt_reverse_count_r2: u32,
    ref_forward_count: u32,
    ref_forward_count_r1: u32,
    ref_forward_count_r2: u32,
    ref_reverse_count: u32,
    ref_reverse_count_r1: u32,
    ref_reverse_count_r2: u32,
    model_probability: f64,
    strand_bias: f64,
}

impl Variant {
    /// Infer the type of variant based on reference and alternate alleles
    ///
    /// # Returns
    /// A string representing the variant type (e.g., "SNP", "INS", "DEL", "MNP", "COMPLEX")
    fn infer_variant_type(&self) -> &'static str {
        let rlen = self.reference.len();
        let alen = self.alt.len();
        match (rlen, alen) {
            (1, 1) => "SNP",
            (r, a) if r > 1 && a > 1 && r == a => "MNP",
            (r, 1) if r > 1 => "DEL",
            (1, a) if a > 1 => "INS",
            _ => "COMPLEX",
        }
    }

    fn tnc_display(&self) -> String {
        if self.infer_variant_type() == "SNP" {
            format!(
                "{}({}>{}){}",
                self.tnc.upstream_base as char,
                self.tnc.ref_base as char,
                self.alt.as_bytes()[0] as char,
                self.tnc.downstream_base as char,
            )
        } else {
            format!(
                "{}{}{}",
                self.tnc.upstream_base as char,
                self.tnc.ref_base as char,
                self.tnc.downstream_base as char,
            )
        }
    }

    /// Render this variant as a VCF record line (newline-terminated).
    fn to_vcf(&self) -> String {
        self.to_vcf_for_mode(&SequencingMode::PairedEnd)
    }

    fn to_vcf_for_mode(&self, sequencing_mode: &SequencingMode) -> String {
        let cd = match self.calling_directive {
            CallingDirective::ReferenceSiteOb => "REF_OB",
            CallingDirective::DenovoSiteOb => "DENOVO_OB",
            CallingDirective::ReferenceSiteOt => "REF_OT",
            CallingDirective::DenovoSiteOt => "DENOVO_OT",
            CallingDirective::BothStrands | CallingDirective::Indel => "BOTH",
        };

        // Clamp zero evidence scores to a small floor so downstream tools can
        // take log without hitting -inf.
        let prob = self.probability.max(1e-300);
        let fwd_prob = self.fwd_probability.max(1e-300);
        let rev_prob = self.rev_probability.max(1e-300);
        let alt_total = self.alt_forward_count + self.alt_reverse_count;

        match sequencing_mode {
            SequencingMode::PairedEnd => format!(
                "{chrom}\t{pos}\t.\t{ref}\t{alt}\t{qual}\t.\tVT={vt};CD={cd};LRP={lrp:.4}\t\
GT:DP:AO:ER:TNC:PR:AMQR:AMQA:AMQR_R1:AMQR_R2:AMQA_R1:AMQA_R2:ABQR:ABQA:ABQR_R1:ABQR_R2:ABQA_R1:ABQA_R2:\
REDR:REDA:REDR_R1:REDR_R2:REDA_R1:REDA_R2:ISR:ISA:FWDP:REVP:LLE:SLE:AMPR:AMPR_R1:AMPR_R2:ARL:ARL_R1:ARL_R2:\
FWD:REV:TOT:FWD_R1:FWD_R2:REV_R1:REV_R2:TOT_R1:TOT_R2:STB\t\
{gt}:{dp}:{ao}:{er:.3E}:{tnc}:{pr:.3E}:\
{amqr:.1}:{amqa:.1}:{amqr_r1:.1}:{amqr_r2:.1}:{amqa_r1:.1}:{amqa_r2:.1}:{abqr:.1}:{abqa:.1}:{abqr_r1:.1}:{abqr_r2:.1}:{abqa_r1:.1}:{abqa_r2:.1}:\
{redr:.1}:{reda:.1}:{redr_r1:.1}:{redr_r2:.1}:{reda_r1:.1}:{reda_r2:.1}:{isr:.1}:{isa:.1}:{fwdp:.3E}:{revp:.3E}:{lle:.3}:{sle:.1}:{ampr:.1}:{ampr_r1:.1}:{ampr_r2:.1}:{arl:.1}:{arl_r1:.1}:{arl_r2:.1}:\
{fwd:.1}:{rev:.1}:{tot:.1}:{fwd_r1:.1}:{fwd_r2:.1}:{rev_r1:.1}:{rev_r2:.1}:{tot_r1:.1}:{tot_r2:.1}:{stb:.3E}\n",
                chrom = self.contig,
                pos   = self.pos,
                ref   = self.reference,
                alt   = self.alt,
                qual  = self.score.round(),
                vt    = self.infer_variant_type(),
                cd    = cd,
                lrp   = self.model_probability,
                gt    = self.genotype,
                dp    = self.depth,
                ao    = self.alt_counts,
                er    = self.error_rate,
                tnc   = self.tnc_display(),
                pr    = prob,
                amqr  = self.average_ref_mapq,
                amqa  = self.average_alt_mapq,
                amqr_r1 = self.average_ref_mapq_r1,
                amqr_r2 = self.average_ref_mapq_r2,
                amqa_r1 = self.average_alt_mapq_r1,
                amqa_r2 = self.average_alt_mapq_r2,
                abqr  = self.average_ref_bq,
                abqa  = self.average_alt_bq,
                abqr_r1 = self.average_ref_bq_r1,
                abqr_r2 = self.average_ref_bq_r2,
                abqa_r1 = self.average_alt_bq_r1,
                abqa_r2 = self.average_alt_bq_r2,
                redr  = self.avg_ref_dist_from_read_end,
                reda  = self.avg_alt_dist_from_read_end,
                redr_r1 = self.avg_ref_dist_from_read_end_r1,
                redr_r2 = self.avg_ref_dist_from_read_end_r2,
                reda_r1 = self.avg_alt_dist_from_read_end_r1,
                reda_r2 = self.avg_alt_dist_from_read_end_r2,
                isr   = self.avg_ref_insert_size,
                isa   = self.avg_alt_insert_size,
                fwdp  = fwd_prob,
                revp  = rev_prob,
                lle   = self.large_local_entropy,
                sle   = self.small_local_entropy,
                ampr  = self.avg_mismatch_per_read,
                ampr_r1 = self.avg_mismatch_per_read_r1,
                ampr_r2 = self.avg_mismatch_per_read_r2,
                arl   = self.avg_read_length,
                arl_r1 = self.avg_read_length_r1,
                arl_r2 = self.avg_read_length_r2,
                fwd   = self.alt_forward_count,
                rev   = self.alt_reverse_count,
                tot   = alt_total,
                fwd_r1 = self.alt_forward_count_r1,
                fwd_r2 = self.alt_forward_count_r2,
                rev_r1 = self.alt_reverse_count_r1,
                rev_r2 = self.alt_reverse_count_r2,
                tot_r1 = self.alt_forward_count_r1 + self.alt_reverse_count_r1,
                tot_r2 = self.alt_forward_count_r2 + self.alt_reverse_count_r2,
                stb   = self.strand_bias,
            ),
            SequencingMode::SingleEnd => format!(
                "{chrom}\t{pos}\t.\t{ref}\t{alt}\t{qual}\t.\tVT={vt};CD={cd};LRP={lrp:.4}\t\
GT:DP:AO:ER:TNC:PR:AMQR:AMQA:ABQR:ABQA:REDR:REDA:ISR:ISA:FWDP:REVP:LLE:SLE:AMPR:ARL:FWD:REV:TOT:STB\t\
{gt}:{dp}:{ao}:{er:.3E}:{tnc}:{pr:.3E}:{amqr:.1}:{amqa:.1}:{abqr:.1}:{abqa:.1}:{redr:.1}:{reda:.1}:{isr:.1}:{isa:.1}:{fwdp:.3E}:{revp:.3E}:{lle:.3}:{sle:.1}:{ampr:.1}:{arl:.1}:{fwd:.1}:{rev:.1}:{tot:.1}:{stb:.3E}\n",
                chrom = self.contig,
                pos   = self.pos,
                ref   = self.reference,
                alt   = self.alt,
                qual  = self.score.round(),
                vt    = self.infer_variant_type(),
                cd    = cd,
                lrp   = self.model_probability,
                gt    = self.genotype,
                dp    = self.depth,
                ao    = self.alt_counts,
                er    = self.error_rate,
                tnc   = self.tnc_display(),
                pr    = prob,
                amqr  = self.average_ref_mapq,
                amqa  = self.average_alt_mapq,
                abqr  = self.average_ref_bq,
                abqa  = self.average_alt_bq,
                redr  = self.avg_ref_dist_from_read_end,
                reda  = self.avg_alt_dist_from_read_end,
                isr   = self.avg_ref_insert_size,
                isa   = self.avg_alt_insert_size,
                fwdp  = fwd_prob,
                revp  = rev_prob,
                lle   = self.large_local_entropy,
                sle   = self.small_local_entropy,
                ampr  = self.avg_mismatch_per_read,
                arl   = self.avg_read_length,
                fwd   = self.alt_forward_count,
                rev   = self.alt_reverse_count,
                tot   = alt_total,
                stb   = self.strand_bias,
            ),
        }
    }
}

/// Representation of a genotype with associated quality score
///
/// # Fields
/// * `genotype` - Genotype string (e.g., "0/1")
/// * `score` - Phred-scaled quality score
struct Genotype {
    genotype: String,
    score: f64,
}

impl Genotype {
    /// Create a new Genotype instance with phred-scaled quality score
    ///
    /// # Arguments
    /// * `genotype` - Genotype string (e.g., "0/1")
    /// * `best_prob` - Probability of the best genotype
    /// * `all_probs_sum` - Sum of probabilities of all genotypes
    ///
    /// # Returns
    /// A new Genotype instance with calculated quality score
    fn new(genotype: &str, best_prob: f64, all_probs_sum: f64) -> Self {
        let p_best = best_prob / all_probs_sum;
        let p_err = (1.0 - p_best).max(1e-300);
        let score = (-10.0 * p_err.log10()).min(999.0);
        Genotype { genotype: genotype.to_string(), score }
    }
}

#[derive(Clone, Debug)]
/// Calling directives for the Taps Variant Caller
///# Variants
/// * `ReferenceSiteOb` - Call at reference site on original bottom strand
/// * `DenovoSiteOb` - Call at de novo site on original bottom strand
/// * `ReferenceSiteOt` - Call at reference site on original top strand
/// * `DenovoSiteOt` - Call at de novo site on original top strand
/// * `BothStrands` - Call on both strands
enum CallingDirective {
    ReferenceSiteOb,
    DenovoSiteOb,
    ReferenceSiteOt,
    DenovoSiteOt,
    BothStrands,
    Indel,
}
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
/// Types of variant observations
///
/// # Variants
/// * `Snp` - Single nucleotide polymorphism
/// * `Insertion` - Insertion variant
/// * `Deletion` - Deletion variant
/// * `Ref` - Reference allele
enum VariantObservation {
    Snp,
    Insertion,
    Deletion,
    Ref,
}

#[derive(Clone, Debug)]
/// Representation of a base call from a read alignment
///
/// # Fields
/// * `base` - The base called from the read
/// * `ref_base` - The reference base at the position
/// * `deleted_bases` - Bases deleted in the read
/// * `insertion_bases` - Bases inserted in the read
struct BaseCall {
    base: char,
    ref_base: char,
    deleted_bases: Vec<u8>,
    insertion_bases: Vec<u8>,
}

impl BaseCall {
    /// Create a new BaseCall instance from an alignment
    ///
    /// # Arguments
    /// * `alignment` - The pileup alignment
    /// * `ref_seq` - The reference sequence as a byte vector
    /// * `ref_pos` - The reference position
    ///
    /// # Returns
    /// A new BaseCall instance
    fn new(alignment: &Alignment, ref_seq: &[u8], ref_pos: u32) -> Self {
        let qpos = alignment.qpos().unwrap();
        let base = alignment.record().seq().as_bytes()[qpos] as char;
        let ref_base = ref_seq[ref_pos as usize] as char;

        let mut deleted_bases = Vec::new();
        let mut insertion_bases = Vec::new();

        match alignment.indel() {
            Indel::Del(len) => {
                let start = ref_pos as usize + 1;
                let end = start + len as usize;
                deleted_bases = ref_seq.get(start..end).unwrap_or(&[]).to_vec();
            }
            Indel::Ins(len) => {
                let read_seq = alignment.record().seq().as_bytes();
                let start = qpos + 1;
                let end = start + len as usize;
                insertion_bases = read_seq.get(start..end).unwrap_or(&[]).to_vec();
            }
            Indel::None => {}
        }

        BaseCall {
            base,
            ref_base,
            deleted_bases,
            insertion_bases,
        }
    }

    fn check_variant_type(&self) -> VariantObservation {
        if !self.insertion_bases.is_empty() {
            VariantObservation::Insertion
        } else if !self.deleted_bases.is_empty() {
            VariantObservation::Deletion
        } else if self.ref_base != self.base {
            VariantObservation::Snp
        } else {
            VariantObservation::Ref
        }
    }

    /// Get the reference allele string
    ///
    /// # Returns
    /// A string representing the reference allele
    fn get_reference_allele(&self) -> String {
        let mut ref_allele = String::new();
        ref_allele.push(self.ref_base);
        if !self.deleted_bases.is_empty() {
            ref_allele.push_str(&String::from_utf8_lossy(&self.deleted_bases));
        }
        ref_allele
    }

    /// Get the alternate allele string
    ///
    /// # Returns
    /// A string representing the alternate allele
    fn get_alternate_allele(&self) -> String {
        let mut alt_allele = String::new();
        alt_allele.push(self.base);
        if !self.insertion_bases.is_empty() {
            alt_allele.push_str(&String::from_utf8_lossy(&self.insertion_bases));
        }
        alt_allele
    }
}

impl PartialEq for BaseCall {
    /// Compare two BaseCall instances for equality
    ///
    /// # Returns
    /// True if equal, false otherwise
    fn eq(&self, other: &Self) -> bool {
        self.base == other.base
            && self.deleted_bases == other.deleted_bases
            && self.insertion_bases == other.insertion_bases
    }
}

impl Eq for BaseCall {}

impl Hash for BaseCall {
    /// Hash the BaseCall instance
    ///
    /// # Returns
    /// A hash value for the BaseCall
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.base.hash(state);
        self.deleted_bases.hash(state);
        self.insertion_bases.hash(state);
    }
}

// ---------------------------------------------------------------------------
// Genome and BAM utilities
// ---------------------------------------------------------------------------

/// A chunk of the genome for processing
///
/// # Fields
/// * `contig` - Chromosome or contig name
/// * `start` - Start position of the chunk (0-based)
/// * `end` - End position of the chunk (0-based, exclusive)
struct GenomeChunk {
    contig: String,
    start: u64,
    end: u64,
}

impl GenomeChunk {
    /// Create a new GenomeChunk instance
    ///
    /// # Arguments
    /// * `contig` - Chromosome or contig name
    /// * `start` - Start position of the chunk (0-based)
    /// * `end` - End position of the chunk (0-based, exclusive)
    /// # Returns
    /// A new GenomeChunk instance
    fn new(contig: String, start: u64, end: u64) -> Self {
        GenomeChunk { contig, start, end }
    }
}

/// Divide the genome into chunks for processing
///
/// # Arguments
/// * `fasta_path` - Path to the reference FASTA file
/// * `chunk_size` - Size of each chunk
///
/// # Returns
/// A vector of GenomeChunk instances
fn get_genome_chunks(fasta_path: &str, chunk_size: u64) -> Vec<GenomeChunk> {
    let reader = faidx::Reader::from_path(fasta_path).expect("Failed to open FASTA file");
    let seq_names = reader.seq_names().expect("Failed to get sequence names");

    let mut chunks = Vec::new();
    for seq_name in seq_names {
        let seq_len = reader.fetch_seq_len(&seq_name);
        let mut start = 0;
        while start < seq_len {
            let end = (start + chunk_size).min(seq_len);
            chunks.push(GenomeChunk::new(seq_name.clone(), start, end));
            start += chunk_size;
        }
    }
    chunks
}

/// Validate that the FAI and BAM headers have matching contigs and lengths
///
/// # Arguments
/// * `fasta_path` - Path to the reference FASTA file
/// * `bam_path` - Path to the BAM file
///
/// # Returns
/// Ok(()) if validation passes, error otherwise
fn validate_fai_and_bam(
    fasta_path: &str,
    bam_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let fai_reader = faidx::Reader::from_path(fasta_path)?;
    let bam_reader = bam::Reader::from_path(bam_path)?;
    let fai_contigs: HashMap<String, u64> = fai_reader
        .seq_names()?
        .iter()
        .map(|name| {
            let len = fai_reader.fetch_seq_len(name);
            (name.clone(), len)
        })
        .collect();
    let bam_header = bam_reader.header();
    for tid in 0..bam_header.target_count() {
        let name = std::str::from_utf8(bam_header.tid2name(tid))?.to_string();
        let len = bam_header.target_len(tid).unwrap();
        match fai_contigs.get(&name) {
            Some(&fai_len) => {
                if fai_len != len {
                    return Err(format!(
                        "Length mismatch for contig {}: FAI length = {}, BAM length = {}",
                        name, fai_len, len
                    )
                    .into());
                }
            }
            None => {
                return Err(format!("Contig {} found in BAM header but not in FAI", name).into());
            }
        }
    }
    Ok(())
}
// ---------------------------------------------------------------------------
// Variant selection and VCF output
// ---------------------------------------------------------------------------

/// Determine the calling directive based on reference and alternate bases
///
/// # Arguments
/// * `ref_base` - Reference base at the position
/// * `alt_candidates` - Set of alternate base candidates
/// * `upstream_base` - Base upstream of the position
/// * `downstream_base` - Base downstream of the position
///
/// # Returns
/// A CallingDirective indicating where to call variants
fn find_where_to_call_variants(
    ref_base: char,
    alt_candidates: &HashSet<BaseCall>,
    upstream_base: char,
    downstream_base: char,
) -> CallingDirective {
    if alt_candidates.iter().any(|bc| {
        matches!(
            bc.check_variant_type(),
            VariantObservation::Insertion | VariantObservation::Deletion
        )
    }) {
        return CallingDirective::Indel;
    }

    let alt_candidate_bases: HashSet<char> = alt_candidates.iter().map(|bc| bc.base).collect();

    if ref_base == 'C' && downstream_base == 'G' {
        CallingDirective::ReferenceSiteOb
    } else if alt_candidate_bases.contains(&'C') && downstream_base == 'G' {
        CallingDirective::DenovoSiteOb
    } else if ref_base == 'G' && upstream_base == 'C' {
        CallingDirective::ReferenceSiteOt
    } else if alt_candidate_bases.contains(&'G') && upstream_base == 'C' {
        CallingDirective::DenovoSiteOt
    } else {
        CallingDirective::BothStrands
    }
}

/// Generate the VCF header string based on the BAM header
///
/// # Arguments
/// * `header` - The BAM header view
///
/// # Returns
/// A string representing the VCF header
fn get_vcf_header(header: &bam::HeaderView) -> String {
    get_vcf_header_for_mode(header, &SequencingMode::PairedEnd)
}

fn get_vcf_header_for_mode(header: &bam::HeaderView, sequencing_mode: &SequencingMode) -> String {
    let contigs = header
        .target_names()
        .iter()
        .map(|name| {
            let name_str = std::str::from_utf8(name).unwrap();
            let length = header.target_len(header.tid(name).unwrap()).unwrap();
            format!("##contig=<ID={},length={}>", name_str, length)
        })
        .collect::<Vec<_>>()
        .join("\n");

    let format_lines = match sequencing_mode {
        SequencingMode::PairedEnd => concat!(
            "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n",
            "##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Read Depth\">\n",
            "##FORMAT=<ID=AO,Number=1,Type=Integer,Description=\"Alternate Allele Count\">\n",
            "##FORMAT=<ID=ER,Number=1,Type=Float,Description=\"Estimated Error Rate\">\n",
            "##FORMAT=<ID=TNC,Number=1,Type=String,Description=\"Trinucleotide Context (SNPs: up(ref>alt)down; non-SNPs: upstream,ref,downstream)\">\n",
            "##FORMAT=<ID=PR,Number=1,Type=Float,Description=\"Aggregate right-tail binomial evidence score for the emitted call\">\n",
            "##FORMAT=<ID=AMQR,Number=1,Type=Float,Description=\"Average mapping quality of reads supporting the reference allele\">\n",
            "##FORMAT=<ID=AMQA,Number=1,Type=Float,Description=\"Average mapping quality of reads supporting the alternate allele\">\n",
            "##FORMAT=<ID=AMQR_R1,Number=1,Type=Float,Description=\"Average mapping quality of reference-supporting R1 reads\">\n",
            "##FORMAT=<ID=AMQR_R2,Number=1,Type=Float,Description=\"Average mapping quality of reference-supporting R2 reads\">\n",
            "##FORMAT=<ID=AMQA_R1,Number=1,Type=Float,Description=\"Average mapping quality of alternate-supporting R1 reads\">\n",
            "##FORMAT=<ID=AMQA_R2,Number=1,Type=Float,Description=\"Average mapping quality of alternate-supporting R2 reads\">\n",
            "##FORMAT=<ID=ABQR,Number=1,Type=Float,Description=\"Average base quality of reads supporting the reference allele\">\n",
            "##FORMAT=<ID=ABQA,Number=1,Type=Float,Description=\"Average base quality of reads supporting the alternate allele\">\n",
            "##FORMAT=<ID=ABQR_R1,Number=1,Type=Float,Description=\"Average base quality of reference-supporting R1 reads\">\n",
            "##FORMAT=<ID=ABQR_R2,Number=1,Type=Float,Description=\"Average base quality of reference-supporting R2 reads\">\n",
            "##FORMAT=<ID=ABQA_R1,Number=1,Type=Float,Description=\"Average base quality of alternate-supporting R1 reads\">\n",
            "##FORMAT=<ID=ABQA_R2,Number=1,Type=Float,Description=\"Average base quality of alternate-supporting R2 reads\">\n",
            "##FORMAT=<ID=REDR,Number=1,Type=Float,Description=\"Average distance from read end for reads supporting the reference allele\">\n",
            "##FORMAT=<ID=REDA,Number=1,Type=Float,Description=\"Average distance from read end for reads supporting the alternate allele\">\n",
            "##FORMAT=<ID=REDR_R1,Number=1,Type=Float,Description=\"Average distance from read end for reference-supporting R1 reads\">\n",
            "##FORMAT=<ID=REDR_R2,Number=1,Type=Float,Description=\"Average distance from read end for reference-supporting R2 reads\">\n",
            "##FORMAT=<ID=REDA_R1,Number=1,Type=Float,Description=\"Average distance from read end for alternate-supporting R1 reads\">\n",
            "##FORMAT=<ID=REDA_R2,Number=1,Type=Float,Description=\"Average distance from read end for alternate-supporting R2 reads\">\n",
            "##FORMAT=<ID=ISR,Number=1,Type=Float,Description=\"Average insert size for reads supporting the reference allele\">\n",
            "##FORMAT=<ID=ISA,Number=1,Type=Float,Description=\"Average insert size for reads supporting the alternate allele\">\n",
            "##FORMAT=<ID=FWDP,Number=1,Type=Float,Description=\"Fraction of strand-separated aggregate evidence contributed by forward reads\">\n",
            "##FORMAT=<ID=REVP,Number=1,Type=Float,Description=\"Fraction of strand-separated aggregate evidence contributed by reverse reads\">\n",
            "##FORMAT=<ID=LLE,Number=1,Type=Float,Description=\"Large local sequence entropy (50 bp on either side)\">\n",
            "##FORMAT=<ID=SLE,Number=1,Type=Float,Description=\"Small local sequence entropy (15 bp on either side)\">\n",
            "##FORMAT=<ID=AMPR,Number=1,Type=Float,Description=\"Average mismatches per read at the position\">\n",
            "##FORMAT=<ID=AMPR_R1,Number=1,Type=Float,Description=\"Average mismatches per read for R1 reads at the position\">\n",
            "##FORMAT=<ID=AMPR_R2,Number=1,Type=Float,Description=\"Average mismatches per read for R2 reads at the position\">\n",
            "##FORMAT=<ID=ARL,Number=1,Type=Float,Description=\"Average read length of reads covering the position\">\n",
            "##FORMAT=<ID=ARL_R1,Number=1,Type=Float,Description=\"Average read length for R1 reads covering the position\">\n",
            "##FORMAT=<ID=ARL_R2,Number=1,Type=Float,Description=\"Average read length for R2 reads covering the position\">\n",
            "##FORMAT=<ID=FWD,Number=1,Type=Float,Description=\"Alternate-supporting forward-strand read count\">\n",
            "##FORMAT=<ID=FWD_R1,Number=1,Type=Float,Description=\"Alternate-supporting forward-strand R1 read count\">\n",
            "##FORMAT=<ID=FWD_R2,Number=1,Type=Float,Description=\"Alternate-supporting forward-strand R2 read count\">\n",
            "##FORMAT=<ID=REV,Number=1,Type=Float,Description=\"Alternate-supporting reverse-strand read count\">\n",
            "##FORMAT=<ID=REV_R1,Number=1,Type=Float,Description=\"Alternate-supporting reverse-strand R1 read count\">\n",
            "##FORMAT=<ID=REV_R2,Number=1,Type=Float,Description=\"Alternate-supporting reverse-strand R2 read count\">\n",
            "##FORMAT=<ID=TOT,Number=1,Type=Float,Description=\"Total alternate-supporting read count across both strands\">\n",
            "##FORMAT=<ID=TOT_R1,Number=1,Type=Float,Description=\"Total alternate-supporting R1 read count\">\n",
            "##FORMAT=<ID=TOT_R2,Number=1,Type=Float,Description=\"Total alternate-supporting R2 read count\">\n",
            "##FORMAT=<ID=STB,Number=1,Type=Float,Description=\"Fisher exact strand-bias p-value for reference versus alternate support\">\n"
        ),
        SequencingMode::SingleEnd => concat!(
            "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n",
            "##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Read Depth\">\n",
            "##FORMAT=<ID=AO,Number=1,Type=Integer,Description=\"Alternate Allele Count\">\n",
            "##FORMAT=<ID=ER,Number=1,Type=Float,Description=\"Estimated Error Rate\">\n",
            "##FORMAT=<ID=TNC,Number=1,Type=String,Description=\"Trinucleotide Context (SNPs: up(ref>alt)down; non-SNPs: upstream,ref,downstream)\">\n",
            "##FORMAT=<ID=PR,Number=1,Type=Float,Description=\"Aggregate right-tail binomial evidence score for the emitted call\">\n",
            "##FORMAT=<ID=AMQR,Number=1,Type=Float,Description=\"Average mapping quality of reads supporting the reference allele\">\n",
            "##FORMAT=<ID=AMQA,Number=1,Type=Float,Description=\"Average mapping quality of reads supporting the alternate allele\">\n",
            "##FORMAT=<ID=ABQR,Number=1,Type=Float,Description=\"Average base quality of reads supporting the reference allele\">\n",
            "##FORMAT=<ID=ABQA,Number=1,Type=Float,Description=\"Average base quality of reads supporting the alternate allele\">\n",
            "##FORMAT=<ID=REDR,Number=1,Type=Float,Description=\"Average distance from read end for reads supporting the reference allele\">\n",
            "##FORMAT=<ID=REDA,Number=1,Type=Float,Description=\"Average distance from read end for reads supporting the alternate allele\">\n",
            "##FORMAT=<ID=ISR,Number=1,Type=Float,Description=\"Average insert size for reads supporting the reference allele\">\n",
            "##FORMAT=<ID=ISA,Number=1,Type=Float,Description=\"Average insert size for reads supporting the alternate allele\">\n",
            "##FORMAT=<ID=FWDP,Number=1,Type=Float,Description=\"Fraction of strand-separated aggregate evidence contributed by forward reads\">\n",
            "##FORMAT=<ID=REVP,Number=1,Type=Float,Description=\"Fraction of strand-separated aggregate evidence contributed by reverse reads\">\n",
            "##FORMAT=<ID=LLE,Number=1,Type=Float,Description=\"Large local sequence entropy (50 bp on either side)\">\n",
            "##FORMAT=<ID=SLE,Number=1,Type=Float,Description=\"Small local sequence entropy (15 bp on either side)\">\n",
            "##FORMAT=<ID=AMPR,Number=1,Type=Float,Description=\"Average mismatches per read at the position\">\n",
            "##FORMAT=<ID=ARL,Number=1,Type=Float,Description=\"Average read length of reads covering the position\">\n",
            "##FORMAT=<ID=FWD,Number=1,Type=Float,Description=\"Alternate-supporting forward-strand read count\">\n",
            "##FORMAT=<ID=REV,Number=1,Type=Float,Description=\"Alternate-supporting reverse-strand read count\">\n",
            "##FORMAT=<ID=TOT,Number=1,Type=Float,Description=\"Total alternate-supporting read count across both strands\">\n",
            "##FORMAT=<ID=STB,Number=1,Type=Float,Description=\"Fisher exact strand-bias p-value for reference versus alternate support\">\n"
        ),
    };

    format!(
        "##fileformat=VCFv4.3\n{}\n##INFO=<ID=VT,Number=1,Type=String,Description=\"Variant Type\">\n##INFO=<ID=CD,Number=1,Type=String,Description=\"TVC Call Directive\">\n##INFO=<ID=LRP,Number=1,Type=Float,Description=\"ML model probability for this call\">\n{}#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample\n",
        contigs,
        format_lines,
    )
}

/// Calculate the right-tail p-value for a binomial distribution
///
/// # Arguments
/// * `n` - Number of trials
/// * `k` - Number of successes
/// * `p` - Probability of success on each trial
///
/// # Returns
/// Right-tail p-value
fn right_tail_binomial_pval(n: u64, k: u64, p: f64) -> f64 {
    let binom = Binomial::new(p, n).expect("Failed to create binomial dist");
    let cdf = binom.cdf(k - 1);
    1.0 - cdf
}

fn log_binom(n: u64, k: u64) -> f64 {
    let k = k.min(n.saturating_sub(k));
    let mut result = 0.0;
    for i in 1..=k {
        result += ((n - k + i) as f64).ln() - (i as f64).ln();
    }
    result
}

fn fisher_exact_test(a: u64, b: u64, c: u64, d: u64) -> f64 {
    let row1 = a + b;
    let row2 = c + d;
    let col1 = a + c;
    let total = row1 + row2;

    if total == 0 || row1 == 0 || row2 == 0 || col1 == 0 || total < col1 {
        return 1.0;
    }

    let observed = (log_binom(row1, a) + log_binom(row2, c) - log_binom(total, col1)).exp();
    let x_min = col1.saturating_sub(row2).max(0);
    let x_max = row1.min(col1);

    let mut p_value = 0.0;
    for x in x_min..=x_max {
        let y = col1 - x;
        if x > row1 || y > row2 {
            continue;
        }
        let prob = (log_binom(row1, x) + log_binom(row2, y) - log_binom(total, col1)).exp();
        if prob <= observed + 1e-12 {
            p_value += prob;
        }
    }

    p_value.min(1.0).max(0.0)
}

fn strand_bias_fisher_pvalue(variant: &Variant) -> f64 {
    let alt_fwd = variant.alt_forward_count as u64;
    let alt_rev = variant.alt_reverse_count as u64;
    let ref_fwd = variant.ref_forward_count as u64;
    let ref_rev = variant.ref_reverse_count as u64;

    if alt_fwd + alt_rev == 0 {
        return 1.0;
    }

    fisher_exact_test(alt_fwd, ref_fwd, alt_rev, ref_rev)
}

fn read_number_of(record: &bam::Record, sequencing_mode: &SequencingMode) -> ReadNumber {
    match sequencing_mode {
        SequencingMode::SingleEnd => ReadNumber::R1,
        SequencingMode::PairedEnd => {
            if record.is_last_in_template() {
                ReadNumber::R2
            } else {
                ReadNumber::R1
            }
        }
    }
}

fn get_count_vec_candidates(
    counts: &HashMap<BaseCall, usize>,
    error_rate: f64,
) -> (HashSet<BaseCall>, Vec<f64>) {
    let total_depth = counts.values().sum::<usize>() as u64;
    let mut candidates = HashSet::new();
    let mut probabilities = Vec::new();

    for (basecall, &count) in counts {
        let variant = basecall.check_variant_type();
        let probability = right_tail_binomial_pval(total_depth, count as u64, error_rate);

        let keep = match variant {
            VariantObservation::Snp if basecall.base == 'N' || basecall.base == basecall.ref_base => false,
            VariantObservation::Insertion | VariantObservation::Deletion if basecall.base == 'N' => false,
            VariantObservation::Ref => false,
            _ => true,
        };

        if keep {
            candidates.insert(basecall.clone());
            probabilities.push(probability);
        }
    }

    (candidates, probabilities)
}

/// Assign genotype based on binomial probabilities
///
/// # Arguments
/// * `alt_counts` - Count of reads supporting the alternate allele
/// * `depth` - Total read depth at the position
/// * `error_rate` - Expected error rate
///
/// # Returns
/// A Genotype instance with assigned genotype and quality score
// REVIEW: this doesn't mean the same thing as germline
fn assign_genotype(alt_counts: usize, depth: usize, error_rate: f64) -> Genotype {
    let homo_ref_prob = Binomial::new(error_rate, depth as u64)
        .unwrap()
        .pmf(alt_counts as u64);
    let het_prob = Binomial::new(0.5, depth as u64)
        .unwrap()
        .pmf(alt_counts as u64);
    let homo_alt_prob = Binomial::new(1.0 - error_rate, depth as u64)
        .unwrap()
        .pmf(alt_counts as u64);

    let total = homo_ref_prob + het_prob + homo_alt_prob;

    let (gt, best_prob) = if homo_ref_prob > het_prob && homo_ref_prob > homo_alt_prob {
        ("0/0", homo_ref_prob)
    } else if het_prob > homo_ref_prob && het_prob > homo_alt_prob {
        ("0/1", het_prob)
    } else {
        ("1/1", homo_alt_prob)
    };
    Genotype::new(gt, best_prob, total)
}

// ---------------------------------------------------------------------------
// ML model inference
// ---------------------------------------------------------------------------

#[derive(Debug)]
#[cfg_attr(not(feature = "onnx-inference"), allow(dead_code))]
struct ModelInferenceConfig {
    model_path: String,
    model_exists: bool,
    /// Feature order the model was trained with. Set exactly once from the
    /// model's `feature_order` metadata in the ONNX file. A model without
    /// this metadata is rejected.
    #[cfg(feature = "onnx-inference")]
    feature_order: OnceLock<Vec<String>>,
}

#[cfg(feature = "onnx-inference")]
fn canonical_base(base: char) -> char {
    match base.to_ascii_uppercase() {
        'A' | 'C' | 'G' | 'T' | 'N' => base.to_ascii_uppercase(),
        _ => 'N',
    }
}

fn model_inference_config(model_path: &str) -> &'static ModelInferenceConfig {
    static CONFIG: OnceLock<ModelInferenceConfig> = OnceLock::new();
    CONFIG.get_or_init(|| {
        let model_path = model_path.to_string();
        let model_exists = Path::new(&model_path).exists();

        if model_exists {
            #[cfg(feature = "onnx-inference")]
            info!(
                "Detected ONNX model at {}. Real ONNX inference is enabled. Feature order must be present in the model metadata.",
                model_path
            );

            #[cfg(not(feature = "onnx-inference"))]
            info!(
                "Detected ONNX model at {}. Built without 'onnx-inference' feature, so model scoring is disabled and fallback behavior is active.",
                model_path
            );
        } else {
            info!(
                "No ONNX model found at {}. Baseline caller behavior is active (no ML filtering).",
                model_path
            );
        }

        ModelInferenceConfig {
            model_path,
            model_exists,
            #[cfg(feature = "onnx-inference")]
            feature_order: OnceLock::new(),
        }
    })
}

#[cfg(feature = "onnx-inference")]
fn onnx_inference_enabled(config: &ModelInferenceConfig) -> bool {
    if !config.model_exists {
        return false;
    }

    if cfg!(test) {
        let skip = std::env::var("TVC_SKIP_ORT_IN_TESTS")
            .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false);

        if skip {
            static ONNX_TEST_SKIP_LOGGED: OnceLock<()> = OnceLock::new();
            ONNX_TEST_SKIP_LOGGED.get_or_init(|| {
                info!("Skipping ONNX Runtime inference in tests due to TVC_SKIP_ORT_IN_TESTS=1.");
            });
            return false;
        }
    }

    true
}

#[cfg(feature = "onnx-inference")]
fn ensure_onnx_runtime_initialized() {
    static ORT_INIT: OnceLock<()> = OnceLock::new();
    ORT_INIT.get_or_init(|| {
        let _ = ort::init()
            .with_name("tvc")
            .with_telemetry(false)
            .commit();
    });
}

/// Build model input features keyed by name.
///
/// IMPORTANT: names must stay in sync with `BASE_FEATURE_NAMES` and with model training.
#[cfg(feature = "onnx-inference")]
fn build_model_feature_map(v: &Variant) -> HashMap<String, f64> {
    let depth = v.depth as f64;
    let alt_counts = v.alt_counts as f64;
    let af = if depth > 0.0 { alt_counts / depth } else { 0.0 };
    let strand_bias = strand_bias_fisher_pvalue(v);

    let mut values = HashMap::<String, f64>::new();
    values.insert("DP".to_string(), depth);
    values.insert("AO".to_string(), alt_counts);
    values.insert("ER".to_string(), v.error_rate);
    values.insert("PR".to_string(), v.probability);
    values.insert("AMQR".to_string(), v.average_ref_mapq);
    values.insert("AMQA".to_string(), v.average_alt_mapq);
    values.insert("ABQR".to_string(), v.average_ref_bq);
    values.insert("ABQA".to_string(), v.average_alt_bq);
    values.insert("AMQR_R1".to_string(), v.average_ref_mapq_r1);
    values.insert("AMQR_R2".to_string(), v.average_ref_mapq_r2);
    values.insert("AMQA_R1".to_string(), v.average_alt_mapq_r1);
    values.insert("AMQA_R2".to_string(), v.average_alt_mapq_r2);
    values.insert("ABQR_R1".to_string(), v.average_ref_bq_r1);
    values.insert("ABQR_R2".to_string(), v.average_ref_bq_r2);
    values.insert("ABQA_R1".to_string(), v.average_alt_bq_r1);
    values.insert("ABQA_R2".to_string(), v.average_alt_bq_r2);
    values.insert("REDR_R1".to_string(), v.avg_ref_dist_from_read_end_r1);
    values.insert("REDR_R2".to_string(), v.avg_ref_dist_from_read_end_r2);
    values.insert("REDA_R1".to_string(), v.avg_alt_dist_from_read_end_r1);
    values.insert("REDA_R2".to_string(), v.avg_alt_dist_from_read_end_r2);
    values.insert("ISR".to_string(), v.avg_ref_insert_size);
    values.insert("ISA".to_string(), v.avg_alt_insert_size);
    values.insert("FWDP".to_string(), v.fwd_probability);
    values.insert("REVP".to_string(), v.rev_probability);
    values.insert("LLE".to_string(), v.large_local_entropy);
    values.insert("SLE".to_string(), v.small_local_entropy);
    values.insert("AMPR_R1".to_string(), v.avg_mismatch_per_read_r1);
    values.insert("AMPR_R2".to_string(), v.avg_mismatch_per_read_r2);
    values.insert("ARL_R1".to_string(), v.avg_read_length_r1);
    values.insert("ARL_R2".to_string(), v.avg_read_length_r2);
    values.insert("FWD_R1".to_string(), v.alt_forward_count_r1 as f64);
    values.insert("FWD_R2".to_string(), v.alt_forward_count_r2 as f64);
    values.insert("REV_R1".to_string(), v.alt_reverse_count_r1 as f64);
    values.insert("REV_R2".to_string(), v.alt_reverse_count_r2 as f64);
    values.insert("TOT_R1".to_string(), (v.alt_forward_count_r1 + v.alt_reverse_count_r1) as f64);
    values.insert("TOT_R2".to_string(), (v.alt_forward_count_r2 + v.alt_reverse_count_r2) as f64);
    values.insert("AF".to_string(), af);
    // REVIEW: change this to a fisher's exact test
    values.insert("strand_bias".to_string(), strand_bias);

    let up = canonical_base(v.tnc.upstream_base as char);
    let rf = canonical_base(v.tnc.ref_base as char);
    let dn = canonical_base(v.tnc.downstream_base as char);

    // Training-script compatible pattern seen in exported feature_order:
    // TNC_up_<triplet>, where triplet was parsed from the raw TNC token.
    // REVIEW: consider changing from TNC_up
    for b in MODEL_TNC_BASES {
        for r in MODEL_TNC_BASES {
            for d in MODEL_TNC_BASES {
                values.insert(
                    format!("TNC_up_{}{}{}", b, r, d),
                    if [up, rf, dn] == [b, r, d] { 1.0 } else { 0.0 },
                );
            }
        }
    }

    let vt = v.infer_variant_type();
    for vt_value in MODEL_VT_VALUES {
        values.insert(
            format!("VT_{}", vt_value),
            if vt == vt_value { 1.0 } else { 0.0 },
        );
    }

    values
}

#[cfg(feature = "onnx-inference")]
fn build_model_feature_vector(variant: &Variant, feature_order: &[String]) -> Vec<f32> {
    let values = build_model_feature_map(variant);

    feature_order
        .iter()
        .map(|k| values.get(k).copied().unwrap_or(0.0) as f32)
        .collect()
}

#[cfg(feature = "onnx-inference")]
fn generated_model_feature_keys() -> HashSet<String> {
    let mut keys: HashSet<String> = BASE_FEATURE_NAMES.iter().map(|s| s.to_string()).collect();
    for b in MODEL_TNC_BASES {
        for r in MODEL_TNC_BASES {
            for d in MODEL_TNC_BASES {
                keys.insert(format!("TNC_up_{}{}{}", b, r, d));
            }
        }
    }
    for vt in MODEL_VT_VALUES {
        keys.insert(format!("VT_{}", vt));
    }
    keys
}

#[cfg(feature = "onnx-inference")]
fn validate_model_feature_order(feature_order: &[String]) -> Result<(), String> {
    let generated = generated_model_feature_keys();
    let ordered_set: HashSet<&str> = feature_order.iter().map(|s| s.as_str()).collect();

    let unsupported: Vec<String> = feature_order
        .iter()
        .filter(|k| !generated.contains((*k).as_str()))
        .cloned()
        .collect();

    let unused_generated = generated
        .iter()
        .filter(|k| !ordered_set.contains(k.as_str()))
        .count();

    if !unsupported.is_empty() {
        let sample = unsupported
            .iter()
            .take(12)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        warn!(
            "ONNX feature_order contains {} unsupported features (sample: {}).",
            unsupported.len(),
            sample
        );
    }

    let unsupported_ratio = unsupported.len() as f64 / feature_order.len().max(1) as f64;
    if unsupported_ratio >= 0.05 || unsupported.len() >= 5 {
        return Err(format!(
            "Model feature_order is incompatible with generated feature schema: {} unsupported of {} total features (ratio {:.1}%).",
            unsupported.len(),
            feature_order.len(),
            unsupported_ratio * 100.0
        ));
    }

    if unused_generated > 0 {
        info!(
            "Model feature_order uses {} / {} generated feature keys.",
            feature_order.len() - unsupported.len(),
            generated.len()
        );
    }

    Ok(())
}

#[cfg(feature = "onnx-inference")]
fn load_onnx_session(model_path: &str) -> Option<OrtSession> {
    ensure_onnx_runtime_initialized();

    let mut builder = match OrtSession::builder() {
        Ok(b) => b,
        Err(err) => {
            static ONNX_MODEL_LOAD_WARNING_LOGGED: OnceLock<()> = OnceLock::new();
            ONNX_MODEL_LOAD_WARNING_LOGGED.get_or_init(|| {
                warn!(
                    "Failed to initialize ONNX Runtime session builder ({}). Falling back to baseline scoring.",
                    err
                );
            });
            return None;
        }
    };

    match builder.commit_from_file(model_path) {
        Ok(model) => Some(model),
        Err(err) => {
            static ONNX_MODEL_LOAD_WARNING_LOGGED: OnceLock<()> = OnceLock::new();
            ONNX_MODEL_LOAD_WARNING_LOGGED.get_or_init(|| {
                warn!(
                    "Failed to load ONNX model at {} with ONNX Runtime backend ({}). Falling back to baseline scoring.",
                    model_path,
                    err
                );
            });
            None
        }
    }
}

#[cfg(feature = "onnx-inference")]
fn parse_feature_order_from_metadata(raw: &str) -> Option<Vec<String>> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }

    let mut parsed = Vec::new();

    // Support JSON-array encoded metadata from skl2onnx exporters.
    if trimmed.starts_with('[') && trimmed.ends_with(']') {
        for item in trimmed
            .trim_start_matches('[')
            .trim_end_matches(']')
            .split(',')
        {
            let token = item.trim().trim_matches('"').trim_matches('\'');
            if !token.is_empty() && token.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
                parsed.push(token.to_string());
            }
        }
    } else {
        for item in trimmed.split(',') {
            let token = item.trim();
            if !token.is_empty()
                && !token.contains('=')
                && token.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
            {
                parsed.push(token.to_string());
            }
        }
    }

    if parsed.is_empty() {
        None
    } else {
        Some(parsed)
    }
}

#[cfg(feature = "onnx-inference")]
fn read_feature_order_from_session(session: &OrtSession) -> Option<Vec<String>> {
    let metadata = session.metadata().ok()?;
    for key in ["feature_order", "feature_names"] {
        let raw = metadata.custom(key)?;
        if let Some(feature_order) = parse_feature_order_from_metadata(&raw) {
            return Some(feature_order);
        }
    }

    None
}

#[cfg(feature = "onnx-inference")]
fn expected_input_width(model: &OrtSession) -> Option<usize> {
    let first_input = model.inputs().first()?;
    let shape = first_input.dtype().tensor_shape()?;
    let width = *shape.last()?;
    if width > 0 {
        Some(width as usize)
    } else {
        None
    }
}

/// Determine the model's feature order from its metadata. The ONNX file must
/// include feature_order/feature_names metadata; sidecar files are not used.
#[cfg(feature = "onnx-inference")]
fn resolve_feature_order(
    session: &OrtSession,
    config: &ModelInferenceConfig,
) -> Result<Vec<String>, String> {
    let from_metadata = read_feature_order_from_session(session).ok_or_else(|| {
        format!(
            "No feature order metadata found in model {}: ONNX metadata keys 'feature_order'/'feature_names' are required",
            config.model_path
        )
    })?;

    validate_model_feature_order(&from_metadata).map_err(|err| {
        format!(
            "Invalid ONNX feature metadata (feature_order/feature_names) in {} ({})",
            config.model_path, err
        )
    })?;

    info!(
        "Loaded ONNX feature_order metadata with {} features.",
        from_metadata.len()
    );

    if let Some(expected_width) = expected_input_width(session) {
        if expected_width != from_metadata.len() {
            return Err(format!(
                "Feature order for model {} has {} features but the model expects {}",
                config.model_path,
                from_metadata.len(),
                expected_width
            ));
        }
    }

    Ok(from_metadata)
}

/// Resolve and store the model's feature order up front so that a model
/// without one fails the run instead of being silently mis-scored.
#[cfg(feature = "onnx-inference")]
fn require_model_feature_order(model_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let config = model_inference_config(model_path);
    if !onnx_inference_enabled(config) || config.feature_order.get().is_some() {
        return Ok(());
    }
    // If the session cannot be loaded, load_onnx_session has already warned and
    // scoring falls back to the baseline caller.
    if let Some(session) = load_onnx_session(&config.model_path) {
        let order = resolve_feature_order(&session, config)?;
        let _ = config.feature_order.set(order);
    }
    Ok(())
}

#[cfg(feature = "onnx-inference")]
/// ONNX inference hook.
fn run_onnx_inference(
    model: &mut OrtSession,
    features: &[f32],
) -> Result<f64, Box<dyn std::error::Error>> {
    if features.len() < 2 {
        return Err("Feature vector must contain at least DP and AO".into());
    }

    if let Some(expected_width) = expected_input_width(model) {
        if features.len() != expected_width {
            return Err(format!(
                "Model input width mismatch: model expects {}, got {} features",
                expected_width,
                features.len()
            )
            .into());
        }
    }

    let input = TensorRef::from_array_view(([1_usize, features.len()], features))?;
    let outputs = model.run(ort::inputs![input])?;
    if outputs.len() == 0 {
        return Err("Model returned no outputs".into());
    }

    let pair_to_probability = |a: f64, b: f64| -> f64 {
        let sum = a + b;
        if a >= 0.0
            && a <= 1.0
            && b >= 0.0
            && b <= 1.0
            && (sum - 1.0).abs() < 1e-3
        {
            b
        } else if sum > 0.0 {
            b / sum
        } else {
            0.0
        }
    };

    // Prefer a probability matrix output (e.g. sklearn ONNX [N,2]) and use class-1 probability.
    for (_, output) in &outputs {
        if let Ok((_, values)) = output.try_extract_tensor::<f32>() {
            if values.len() >= 2 {
                return Ok(pair_to_probability(values[0] as f64, values[1] as f64).clamp(0.0, 1.0));
            }
        }
    }

    // Fall back to scalar probability outputs.
    for (_, output) in &outputs {
        if let Ok((_, values)) = output.try_extract_tensor::<f32>() {
            if values.len() == 1 {
                let v = values[0] as f64;
                let p = if (0.0..=1.0).contains(&v) {
                    v
                } else {
                    1.0 / (1.0 + (-v).exp())
                };
                return Ok(p.clamp(0.0, 1.0));
            }
        }
    }

    Err("Could not find a numeric probability output tensor in ONNX outputs".into())
}

/// Score a variant with the ML model. Returns 1.0 (keep the call) whenever no
/// model is available or inference fails.
#[cfg(feature = "onnx-inference")]
fn model_probability_score(config: &ModelInferenceConfig, variant: &Variant) -> f64 {
    if !onnx_inference_enabled(config) {
        // No model file yet: keep baseline caller behavior (no ML filtering).
        return 1.0;
    }

    THREAD_LOCAL_ONNX_MODELS.with(|models| {
        let mut models = models.borrow_mut();
        if !models.contains_key(config.model_path.as_str()) {
            let mut model = load_onnx_session(&config.model_path);
            // Read the feature order from this session before storing it, so we
            // never open a second ORT session (which can deadlock on ORT's
            // internal environment mutex).
            let order_result = match (&model, config.feature_order.get()) {
                (Some(session), None) => Some(resolve_feature_order(session, config)),
                _ => None,
            };
            match order_result {
                Some(Ok(order)) => {
                    // Another thread may have won the race; the order is identical.
                    let _ = config.feature_order.set(order);
                }
                Some(Err(err)) => {
                    warn!("{}. Falling back to baseline scoring.", err);
                    model = None;
                }
                None => {}
            }
            models.insert(config.model_path.clone(), model);
        }

        let Some(model) = models
            .get_mut(config.model_path.as_str())
            .and_then(|m| m.as_mut())
        else {
            return 1.0;
        };
        let Some(feature_order) = config.feature_order.get() else {
            return 1.0;
        };

        let features = build_model_feature_vector(variant, feature_order);
        run_onnx_inference(model, &features).unwrap_or(1.0)
    })
}

/// Without the `onnx-inference` feature no model scoring happens: keep baseline
/// caller behavior (no ML filtering).
#[cfg(not(feature = "onnx-inference"))]
fn model_probability_score(_config: &ModelInferenceConfig, _variant: &Variant) -> f64 {
    1.0
}

// ---------------------------------------------------------------------------
// Pileup processing
// ---------------------------------------------------------------------------

/// Retrieve an NM tag from a record
///
/// # Arguments
/// * `record` - The record to retrieve the Tags value from
///
/// # Returns
/// The value of the NM tag
fn get_nm_tag(record: &bam::Record) -> u32 {
    match record.aux(b"NM") {
        Ok(bam::record::Aux::I8(n)) => n as u32,
        Ok(bam::record::Aux::U8(n)) => n as u32,
        Ok(bam::record::Aux::I16(n)) => n as u32,
        Ok(bam::record::Aux::U16(n)) => n as u32,
        Ok(bam::record::Aux::I32(n)) => n as u32,
        Ok(bam::record::Aux::U32(n)) => n,
        _ => panic!("NM tag missing or invalid"),
    }
}

/// Determine if a record is the stranded read
///
/// # Arguments
/// * `record` - The record to asses
/// * `stranded_read` which read is stranded
///
/// # Returns
/// True if the read is the stranded one
fn is_stranded_read(
    record: &bam::Record,
    stranded_read: &ReadNumber,
    sequencing_mode: &SequencingMode,
) -> bool {
    let read_orientation = read_number_of(record, sequencing_mode);

    read_orientation == *stranded_read
}

#[derive(Debug)]
/// Counts of basecalls in a pileup
struct PileupCounts {
    fwd: HashMap<BaseCall, usize>,
    rev: HashMap<BaseCall, usize>,
    total: HashMap<BaseCall, usize>,
}

impl PileupCounts {
    fn new() -> Self {
        PileupCounts {
            fwd: HashMap::with_capacity(8),
            rev: HashMap::with_capacity(8),
            total: HashMap::with_capacity(8),
        }
    }
}

/// Return `true` if `sequence` contains a repeated unit of length `n` of at
/// least `cutoff` bases at the start or end.
fn has_repeat(sequence: &[u8], n: usize, cutoff: usize) -> bool {
    let len = sequence.len();
    if len < cutoff || n == 0 {
        return false;
    }

    let unit = &sequence[0..n];
    let start_ok = sequence[..cutoff].chunks(n).all(|chunk| chunk == unit);
    if start_ok {
        return true;
    }

    let tail_unit = &sequence[len - n..];
    let end_ok = sequence[len - cutoff..].chunks(n).all(|chunk| chunk == tail_unit);
    end_ok
}

/// Returns true if the read should be filtered out for INDEL calling
/// Filters reads with repeated sequences at the ends or soft-clipping
fn filter_indels(
    sequence: &[u8],
    record: &bam::Record,
    indel_filter_repeat_limit: usize,
    dinuc_cutoff: usize,
) -> bool {
    let homopolymer = has_repeat(sequence, 1, indel_filter_repeat_limit);
    let dinuc = has_repeat(sequence, 2, dinuc_cutoff);
    let soft_clipped = {
        for cigar in record.cigar().iter() {
            if let Cigar::SoftClip(_) = cigar {
                return true;
            }
        }
        false
    };
    homopolymer || dinuc || soft_clipped
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TrinucleotideContext {
    upstream_base: u8,
    ref_base: u8,
    downstream_base: u8,
}

impl TrinucleotideContext {
    fn new(upstream_base: u8, ref_base: u8, downstream_base: u8) -> Self {
        TrinucleotideContext { upstream_base, ref_base, downstream_base }
    }
}

/// Bases immediately upstream and downstream of `pos` (`N` at sequence edges).
fn flanking_bases(ref_seq: &[u8], pos: usize) -> (u8, u8) {
    let upstream = if pos > 0 { ref_seq[pos - 1] } else { b'N' };
    let downstream = if pos + 1 < ref_seq.len() { ref_seq[pos + 1] } else { b'N' };
    (upstream, downstream)
}

/// Calculate Shannon entropy of a sequence
/// Returns 0 for empty sequences, and is based on the frequency of A, C, G, T
/// Non-ACGT characters are ignored in the calculation
/// The formula is: -sum(p_i * log2(p_i)) for each base i, where p_i is the frequency of base i in the sequence
/// The entropy is measured in bits, and higher values indicate more diversity in the sequence
/// The maximum entropy for a sequence of A, C, G, T is 2 bits (when all bases are equally represented)
/// For example, the sequence "ACGT" has an entropy of 2 bits, while "AAAA" has an entropy of 0 bits
fn shannon_entropy(sequence: &[u8]) -> f64 {
    if sequence.is_empty() {
        return 0.0;
    }
    let mut counts = [0u32; 4];
    let mut valid = 0u32;
    for &base in sequence {
        match base {
            b'A' | b'a' => counts[0] += 1,
            b'C' | b'c' => counts[1] += 1,
            b'G' | b'g' => counts[2] += 1,
            b'T' | b't' => counts[3] += 1,
            _ => {} 
        }
        valid += 1;
    }
    if valid == 0 {
        return 0.0;
    }
    let n = valid as f64;
    counts.iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n;
            -p * p.log2()
        })
        .sum()
}

/// Shannon entropy of the reference in a window of `flank` bases on either side of `pos`.
fn flank_entropy(ref_seq: &[u8], pos: usize, flank: usize) -> f64 {
    shannon_entropy(&ref_seq[pos.saturating_sub(flank)..(pos + flank + 1).min(ref_seq.len())])
}

/// All per-position statistics returned by [`compute_pileup_counts`].
struct PileupStats {
    ref_dist_from_read_end: f64,
    alt_dist_from_read_end: f64,
    ref_dist_from_read_end_r1: f64,
    ref_dist_from_read_end_r2: f64,
    alt_dist_from_read_end_r1: f64,
    alt_dist_from_read_end_r2: f64,
    ref_insert_size_sum: f64,
    alt_insert_size_sum: f64,
    total_alt_counts: f64,
    total_ref_counts: f64,
    total_alt_counts_r1: f64,
    total_alt_counts_r2: f64,
    total_ref_counts_r1: f64,
    total_ref_counts_r2: f64,
    count_ref_mapq: f64,
    count_alt_mapq: f64,
    count_ref_mapq_r1: f64,
    count_ref_mapq_r2: f64,
    count_alt_mapq_r1: f64,
    count_alt_mapq_r2: f64,
    count_ref_bq: f64,
    count_alt_bq: f64,
    count_ref_bq_r1: f64,
    count_ref_bq_r2: f64,
    count_alt_bq_r1: f64,
    count_alt_bq_r2: f64,
    total_mismatches: f64,
    total_mismatches_r1: f64,
    total_mismatches_r2: f64,
    total_read_length: f64,
    total_read_length_r1: f64,
    total_read_length_r2: f64,
    read_end_filtered_count_indels: f64,
    indel_offset: u64,
}

/// Per-position averages derived from [`PileupStats`].
struct SiteAverages {
    ref_mapq: f64,
    alt_mapq: f64,
    ref_mapq_r1: f64,
    ref_mapq_r2: f64,
    alt_mapq_r1: f64,
    alt_mapq_r2: f64,
    ref_bq: f64,
    alt_bq: f64,
    ref_bq_r1: f64,
    ref_bq_r2: f64,
    alt_bq_r1: f64,
    alt_bq_r2: f64,
    ref_dist: f64,
    alt_dist: f64,
    ref_dist_r1: f64,
    ref_dist_r2: f64,
    alt_dist_r1: f64,
    alt_dist_r2: f64,
    ref_ins: f64,
    alt_ins: f64,
    mismatch: f64,
    mismatch_r1: f64,
    mismatch_r2: f64,
    read_length: f64,
    read_length_r1: f64,
    read_length_r2: f64,
}

fn safe_div(num: f64, den: f64) -> f64 {
    if num > 0.0 && den > 0.0 { num / den } else { 0.0 }
}

// ---------------------------------------------------------------------------
// Pileup statistics and candidate filtering
// ---------------------------------------------------------------------------

impl PileupStats {
    fn averages(&self) -> SiteAverages {
        let total_reads = self.total_ref_counts + self.total_alt_counts;
        let total_reads_r1 = self.total_ref_counts_r1 + self.total_alt_counts_r1;
        let total_reads_r2 = self.total_ref_counts_r2 + self.total_alt_counts_r2;
        SiteAverages {
            ref_mapq: safe_div(self.count_ref_mapq, self.total_ref_counts),
            alt_mapq: safe_div(self.count_alt_mapq, self.total_alt_counts),
            ref_mapq_r1: safe_div(self.count_ref_mapq_r1, self.total_ref_counts_r1),
            ref_mapq_r2: safe_div(self.count_ref_mapq_r2, self.total_ref_counts_r2),
            alt_mapq_r1: safe_div(self.count_alt_mapq_r1, self.total_alt_counts_r1),
            alt_mapq_r2: safe_div(self.count_alt_mapq_r2, self.total_alt_counts_r2),
            ref_bq: safe_div(self.count_ref_bq, self.total_ref_counts),
            alt_bq: safe_div(self.count_alt_bq, self.total_alt_counts),
            ref_bq_r1: safe_div(self.count_ref_bq_r1, self.total_ref_counts_r1),
            ref_bq_r2: safe_div(self.count_ref_bq_r2, self.total_ref_counts_r2),
            alt_bq_r1: safe_div(self.count_alt_bq_r1, self.total_alt_counts_r1),
            alt_bq_r2: safe_div(self.count_alt_bq_r2, self.total_alt_counts_r2),
            ref_dist: safe_div(self.ref_dist_from_read_end, self.total_ref_counts),
            alt_dist: safe_div(self.alt_dist_from_read_end, self.total_alt_counts),
            ref_dist_r1: safe_div(self.ref_dist_from_read_end_r1, self.total_ref_counts_r1),
            ref_dist_r2: safe_div(self.ref_dist_from_read_end_r2, self.total_ref_counts_r2),
            alt_dist_r1: safe_div(self.alt_dist_from_read_end_r1, self.total_alt_counts_r1),
            alt_dist_r2: safe_div(self.alt_dist_from_read_end_r2, self.total_alt_counts_r2),
            ref_ins: safe_div(self.ref_insert_size_sum, self.total_ref_counts),
            alt_ins: safe_div(self.alt_insert_size_sum, self.total_alt_counts),
            mismatch: safe_div(self.total_mismatches, total_reads),
            mismatch_r1: safe_div(self.total_mismatches_r1, total_reads_r1),
            mismatch_r2: safe_div(self.total_mismatches_r2, total_reads_r2),
            read_length: safe_div(self.total_read_length, total_reads),
            read_length_r1: safe_div(self.total_read_length_r1, total_reads_r1),
            read_length_r2: safe_div(self.total_read_length_r2, total_reads_r2),
        }
    }
}

/// Compute base call counts from a pileup
///
/// # Arguments
/// * `pileup` - The pileup to extract counts from
/// * `min_bq` - Minimum base quality
/// * `min_mapq` - Minimum mapping quality
/// * `end_of_read_cutoff` - End of read cutoff for SNPs
/// * `indel_end_of_read_cutoff` - End of read cutoff for indels
/// * `max_mismatches` - Maximum allowed mismatches in a read
/// * `ref_seq` - The reference sequence as a byte vector
/// * `ref_pos` - The reference position
/// * `indel_filter_repeat_limit` - Homopolymer length for indel read filtering
///   (the dinucleotide cutoff is derived from it, rounded up to even)
///
/// # Returns
/// A PileupStats instance; strand-resolved counts are written into `pileup_counts`
#[allow(clippy::too_many_arguments)]
fn compute_pileup_counts(
    pileup: &Pileup,
    min_bq: usize,
    min_mapq: usize,
    end_of_read_cutoff: usize,
    indel_end_of_read_cutoff: usize,
    max_mismatches: u32,
    ref_seq: &[u8],
    ref_pos: u32,
    stranded_read: &ReadNumber,
    sequencing_mode: &SequencingMode,
    pileup_counts: &mut PileupCounts,
    indel_filter_repeat_limit: usize,
) -> PileupStats {
    pileup_counts.fwd.clear();
    pileup_counts.rev.clear();
    pileup_counts.total.clear();

    let dinuc_cutoff = indel_filter_repeat_limit.next_multiple_of(2);

    let mut stats = PileupStats {
        ref_dist_from_read_end: 0.0,
        alt_dist_from_read_end: 0.0,
        ref_dist_from_read_end_r1: 0.0,
        ref_dist_from_read_end_r2: 0.0,
        alt_dist_from_read_end_r1: 0.0,
        alt_dist_from_read_end_r2: 0.0,
        ref_insert_size_sum: 0.0,
        alt_insert_size_sum: 0.0,
        total_alt_counts: 0.0,
        total_ref_counts: 0.0,
        total_alt_counts_r1: 0.0,
        total_alt_counts_r2: 0.0,
        total_ref_counts_r1: 0.0,
        total_ref_counts_r2: 0.0,
        count_ref_mapq: 0.0,
        count_alt_mapq: 0.0,
        count_ref_mapq_r1: 0.0,
        count_ref_mapq_r2: 0.0,
        count_alt_mapq_r1: 0.0,
        count_alt_mapq_r2: 0.0,
        count_ref_bq: 0.0,
        count_alt_bq: 0.0,
        count_ref_bq_r1: 0.0,
        count_ref_bq_r2: 0.0,
        count_alt_bq_r1: 0.0,
        count_alt_bq_r2: 0.0,
        total_mismatches: 0.0,
        total_mismatches_r1: 0.0,
        total_mismatches_r2: 0.0,
        total_read_length: 0.0,
        total_read_length_r1: 0.0,
        total_read_length_r2: 0.0,
        read_end_filtered_count_indels: 0.0,
        indel_offset: 0,
    };

    for alignment in pileup.alignments() {
        let record = alignment.record();
        let mismatches = get_nm_tag(&record);

        if mismatches > max_mismatches {
            continue;
        }

        stats.total_mismatches += mismatches as f64;

        let qpos = match alignment.qpos() {
            Some(p) => p,
            None => continue,
        };

        if alignment.is_del() || alignment.is_refskip() {
            continue;
        }

        let base = record.seq().as_bytes()[qpos] as char;
        if base == 'N' {
            continue;
        }

        let qual = record.qual()[qpos];
        let mapq = record.mapq();
        let basecall = BaseCall::new(&alignment, ref_seq, ref_pos);
        let variant_type = basecall.check_variant_type();
        let read_number = read_number_of(&record, sequencing_mode);

        match read_number {
            ReadNumber::R1 => stats.total_mismatches_r1 += mismatches as f64,
            ReadNumber::R2 => stats.total_mismatches_r2 += mismatches as f64,
        }

        if qual < min_bq as u8 || mapq < min_mapq as u8 {
            continue;
        }

        let is_ref = variant_type == VariantObservation::Ref;
        if is_ref {
            stats.total_ref_counts += 1.0;
            match read_number {
                ReadNumber::R1 => stats.total_ref_counts_r1 += 1.0,
                ReadNumber::R2 => stats.total_ref_counts_r2 += 1.0,
            }
            stats.count_ref_mapq += mapq as f64;
            match read_number {
                ReadNumber::R1 => stats.count_ref_mapq_r1 += mapq as f64,
                ReadNumber::R2 => stats.count_ref_mapq_r2 += mapq as f64,
            }
            stats.count_ref_bq += qual as f64;
            match read_number {
                ReadNumber::R1 => stats.count_ref_bq_r1 += qual as f64,
                ReadNumber::R2 => stats.count_ref_bq_r2 += qual as f64,
            }
            let dist = std::cmp::min(qpos, record.seq().len() - 1 - qpos) as f64;
            stats.ref_dist_from_read_end += dist;
            match read_number {
                ReadNumber::R1 => stats.ref_dist_from_read_end_r1 += dist,
                ReadNumber::R2 => stats.ref_dist_from_read_end_r2 += dist,
            }
            stats.ref_insert_size_sum += record.insert_size().unsigned_abs() as f64;
        } else {
            stats.total_alt_counts += 1.0;
            match read_number {
                ReadNumber::R1 => stats.total_alt_counts_r1 += 1.0,
                ReadNumber::R2 => stats.total_alt_counts_r2 += 1.0,
            }
            stats.count_alt_mapq += mapq as f64;
            match read_number {
                ReadNumber::R1 => stats.count_alt_mapq_r1 += mapq as f64,
                ReadNumber::R2 => stats.count_alt_mapq_r2 += mapq as f64,
            }
            stats.count_alt_bq += qual as f64;
            match read_number {
                ReadNumber::R1 => stats.count_alt_bq_r1 += qual as f64,
                ReadNumber::R2 => stats.count_alt_bq_r2 += qual as f64,
            }
            let dist = std::cmp::min(qpos, record.seq().len() - 1 - qpos) as f64;
            stats.alt_dist_from_read_end += dist;
            match read_number {
                ReadNumber::R1 => stats.alt_dist_from_read_end_r1 += dist,
                ReadNumber::R2 => stats.alt_dist_from_read_end_r2 += dist,
            }
            stats.alt_insert_size_sum += record.insert_size().unsigned_abs() as f64;
        }

        if record.is_secondary() || record.is_supplementary() || record.is_duplicate() {
            continue;
        }

        stats.total_read_length += record.seq().len() as f64;
        match read_number {
            ReadNumber::R1 => stats.total_read_length_r1 += record.seq().len() as f64,
            ReadNumber::R2 => stats.total_read_length_r2 += record.seq().len() as f64,
        }

        let read_len = record.seq().len();
        match variant_type {
            VariantObservation::Snp => {
                if qpos < end_of_read_cutoff || qpos >= read_len - end_of_read_cutoff {
                    continue;
                }
            }
            VariantObservation::Insertion | VariantObservation::Deletion => {
                if qpos < indel_end_of_read_cutoff || qpos >= read_len - indel_end_of_read_cutoff {
                    stats.read_end_filtered_count_indels += 1.0;
                }
            }
            _ => {}
        }

        // Strand assignment.
        let on_rev = (record.is_reverse() && is_stranded_read(&record, stranded_read, sequencing_mode))
            || (!record.is_reverse() && !is_stranded_read(&record, stranded_read, sequencing_mode));

        if on_rev {
            *pileup_counts.rev.entry(basecall.clone()).or_insert(0) += 1;
        } else {
            *pileup_counts.fwd.entry(basecall.clone()).or_insert(0) += 1;
        }
        *pileup_counts.total.entry(basecall.clone()).or_insert(0) += 1;

        if is_ref {
            let read_seq = record.seq().as_bytes();
            if filter_indels(&read_seq, &record, indel_filter_repeat_limit, dinuc_cutoff) {
                stats.indel_offset += 1;
            }
        }
    }

    stats
}

/// Distributes counts from a pileup map into SNP and INDEL maps
///
/// # Arguments
/// * `pileup_map` - The pileup counts map
/// * `snp_map` - The SNP counts map to populate
/// * `indel_map` - The INDEL counts map to populate
fn distribute_counts(
    pileup_map: &HashMap<BaseCall, usize>,
    snp_map: &mut HashMap<BaseCall, usize>,
    indel_map: &mut HashMap<BaseCall, usize>,
) {
    for (obs, count) in pileup_map {
        match obs.check_variant_type() {
            VariantObservation::Snp | VariantObservation::Ref => {
                snp_map.insert(obs.clone(), *count);
            }
            VariantObservation::Insertion | VariantObservation::Deletion => {
                indel_map.insert(obs.clone(), *count);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// BAM workflow orchestration
// ---------------------------------------------------------------------------

fn make_progress_bar(len: usize, label: &str) -> Result<ProgressBar, Box<dyn std::error::Error>> {
    let pb = ProgressBar::new(len as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(&format!(
                "{{spinner:.green}} [{{elapsed_precise}}] [{{bar:40.cyan/blue}}] {{pos}}/{{len}} {}",
                label
            ))?
            .progress_chars("#>-"),
    );
    Ok(pb)
}

/// Call variants in every chunk in parallel on `pool`, returning all calls.
/// A chunk that fails to process contributes no calls.
fn call_all_chunks(
    pool: &rayon::ThreadPool,
    chunks: &[GenomeChunk],
    bam_path: &str,
    ref_seqs: &HashMap<String, Vec<u8>>,
    args: &Args,
    ml_threshold: f64,
    pb: &ProgressBar,
) -> Vec<Variant> {
    pool.install(|| {
        chunks
            .par_iter()
            .map(|chunk| {
                let variants = call_variants_impl(
                    chunk,
                    bam_path,
                    ref_seqs
                        .get(&chunk.contig)
                        .expect("Contig not found in reference"),
                    args.min_bq,
                    args.min_mapq,
                    args.min_depth,
                    args.end_of_read_cutoff,
                    args.indel_end_of_read_cutoff,
                    args.max_mismatches,
                    args.min_ao,
                    args.error_rate,
                    &args.stranded_read,
                    &args.sequencing_mode,
                    args.indel_filter_repeat_limit,
                    &args.model_path,
                    ml_threshold,
                )
                .unwrap_or_else(|_e| Vec::new());
                pb.inc(1);
                variants
            })
            .flatten()
            .collect()
    })
}

/// Main workflow for variant calling
///
/// Calls variants in the tumor BAM, optionally removes any call also made in
/// the matched normal BAM, and writes the sorted result as a VCF.
///
/// # Returns
/// Ok(()) if workflow completes successfully, error otherwise
fn workflow(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    info!("Starting TVC workflow");
    let ref_path = args.input_ref.as_str();
    let tumor_bam_path = args.input_bam.as_str();
    let matched_normal_bam_path = args.matched_normal_bam.as_deref();

    validate_fai_and_bam(ref_path, tumor_bam_path)?;
    if let Some(normal_bam_path) = matched_normal_bam_path {
        validate_fai_and_bam(ref_path, normal_bam_path)?;
    }

    #[cfg(feature = "onnx-inference")]
    require_model_feature_order(&args.model_path)?;

    info!("Reading reference sequences");
    let ref_reader = faidx::Reader::from_path(ref_path)?;
    let contigs: Vec<String> = ref_reader.seq_names()?;

    let mut seq_name_to_seq = HashMap::<String, Vec<u8>>::new();

    for contig in &contigs {
        let seq_len = ref_reader.fetch_seq_len(contig);
        let ref_seq: Vec<u8> = ref_reader
            .fetch_seq(contig, 0, seq_len as usize)?
            .into_iter()
            .map(|b| b.to_ascii_uppercase())
            .collect();
        seq_name_to_seq.insert(contig.clone(), ref_seq);
    }

    info!("Dividing genome into chunks and getting ready for parallel processing");

    let chunks: Vec<GenomeChunk> = get_genome_chunks(ref_path, args.chunk_size);

    // Rayon thread pool
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.num_threads)
        .build()?;

    let pb = make_progress_bar(chunks.len(), "chunks processed")?;
    let mut all_variants = call_all_chunks(
        &pool,
        &chunks,
        tumor_bam_path,
        &seq_name_to_seq,
        args,
        args.tumor_ml_threshold,
        &pb,
    );
    pb.finish_with_message("Tumor variant calling complete. Wrapping up.");

    if let Some(normal_bam_path) = matched_normal_bam_path {
        info!(
            "Matched normal BAM provided. Calling normal variants with ML threshold {} and using them to filter tumor calls.",
            args.normal_ml_threshold
        );

        let pb_normal = make_progress_bar(chunks.len(), "normal chunks processed")?;
        let normal_variants = call_all_chunks(
            &pool,
            &chunks,
            normal_bam_path,
            &seq_name_to_seq,
            args,
            args.normal_ml_threshold,
            &pb_normal,
        );
        pb_normal.finish_with_message("Normal variant calling complete.");

        let normal_variant_keys: HashSet<(String, u32, String, String)> = normal_variants
            .into_iter()
            .map(|v| (v.contig, v.pos, v.reference, v.alt))
            .collect();

        all_variants.retain(|v| {
            !normal_variant_keys.contains(&(v.contig.clone(), v.pos, v.reference.clone(), v.alt.clone()))
        });
    }

    // Sort all variants by contig and position
    all_variants.sort_by(|a, b| a.contig.cmp(&b.contig).then(a.pos.cmp(&b.pos)));

    // Write to VCF
    let mut vcf_file = File::create(&args.output_vcf)?;
    let header = bam::Reader::from_path(tumor_bam_path)?.header().to_owned();
    vcf_file.write_all(get_vcf_header_for_mode(&header, &args.sequencing_mode).as_bytes())?;

    for variant in all_variants {
        vcf_file.write_all(variant.to_vcf_for_mode(&args.sequencing_mode).as_bytes())?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// BAM traversal and per-site statistics
// ---------------------------------------------------------------------------

/// Compute trinucleotide context-specific error rates for a genome chunk
///
/// # Arguments
/// * `chunk` - The genome chunk to analyze
/// * `bam_path` - Path to the BAM file
/// * `ref_seq` - The reference sequence for the chunk
/// * `min_bq` - Minimum base quality
/// * `min_mapq` - Minimum mapping quality
/// * `min_depth` - Minimum read depth
/// * `end_of_read_cutoff` - End of read cutoff for SNPs
/// * `indel_end_of_read_cutoff` - End of read cutoff for indels
/// * `max_mismatches` - Maximum allowed mismatches in a read
/// * `error_rate` - Expected general error rate
/// * `stranded_read` - Which read is stranded (R1 or R2)
/// * `indel_filter_repeat_limit` - Number of bases for homopolymer/dinucleotide repeat filtering for indels
/// # Returns
/// A HashMap mapping each trinucleotide context to its estimated error rate
fn compute_tnc_error_rates(
    chunk: &GenomeChunk,
    bam_path: &str,
    ref_seq: &[u8],
    min_bq: usize,
    min_mapq: usize,
    min_depth: u32,
    end_of_read_cutoff: usize,
    indel_end_of_read_cutoff: usize,
    max_mismatches: u32,
    error_rate: f64,
    stranded_read: &ReadNumber,
    sequencing_mode: &SequencingMode,
    indel_filter_repeat_limit: usize,
) -> Result<HashMap<TrinucleotideContext, f64>, Box<dyn std::error::Error>> {
    // Pre-populate every possible TNC with zero counts.
    let bases = [b'A', b'C', b'G', b'T'];
    let mut tnc_counts: HashMap<TrinucleotideContext, (f64, f64)> = HashMap::new();
    for &upstream in &bases {
        for &ref_base in &bases {
            for &downstream in &bases {
                tnc_counts.insert(TrinucleotideContext::new(upstream, ref_base, downstream), (0.0, 0.0));
            }
        }
    }

    let mut bam = bam::IndexedReader::from_path(bam_path)?;
    let header = bam.header().to_owned();
    let tid = header.tid(chunk.contig.as_bytes()).ok_or("Contig not found in BAM header")?;
    bam.fetch((tid, chunk.start as i64, chunk.end as i64))?;

    let mut pileup_counts = PileupCounts::new();

    let mut fwd_snps = HashMap::with_capacity(4);
    let mut rev_snps = HashMap::with_capacity(4);
    let mut total_snps = HashMap::with_capacity(4);
    // distribute_counts always fills an indel map; the error-rate estimate ignores it.
    let mut indel_scratch = HashMap::with_capacity(4);

    for result in bam.pileup() {
        let pileup: Pileup = result?;
        let pos = pileup.pos();

        if pileup.depth() < min_depth {
            continue;
        }

        let ref_base = ref_seq[pos as usize];

        compute_pileup_counts(
            &pileup, min_bq, min_mapq, end_of_read_cutoff, indel_end_of_read_cutoff,
            max_mismatches, ref_seq, pos, stranded_read, sequencing_mode, &mut pileup_counts,
            indel_filter_repeat_limit,
        );

        fwd_snps.clear();
        rev_snps.clear();
        total_snps.clear();
        indel_scratch.clear();

        distribute_counts(&pileup_counts.fwd,   &mut fwd_snps,   &mut indel_scratch);
        distribute_counts(&pileup_counts.rev,   &mut rev_snps,   &mut indel_scratch);
        distribute_counts(&pileup_counts.total, &mut total_snps, &mut indel_scratch);

        let (upstream, downstream) = flanking_bases(ref_seq, pos as usize);

        // Same strand selection as calling: the directive decides which strand's counts are used.
        let (fwd_cands, _) = get_count_vec_candidates(&fwd_snps, error_rate);
        let directive = find_where_to_call_variants(
            ref_base as char, &fwd_cands, upstream as char, downstream as char,
        );
        let counts_snps = match directive {
            CallingDirective::ReferenceSiteOb | CallingDirective::DenovoSiteOb => &rev_snps,
            CallingDirective::ReferenceSiteOt | CallingDirective::DenovoSiteOt => &fwd_snps,
            CallingDirective::BothStrands | CallingDirective::Indel => &total_snps,
        };

        let total_ref_snps: u64 = counts_snps
            .iter()
            .filter(|(k, _)| k.check_variant_type() == VariantObservation::Ref)
            .map(|(_, &v)| v as u64)
            .sum();
        let total_alt_snps: u64 = counts_snps
            .iter()
            .filter(|(k, _)| k.check_variant_type() == VariantObservation::Snp)
            .map(|(_, &v)| v as u64)
            .sum();

        let ctx = TrinucleotideContext::new(upstream, ref_base, downstream);
        let entry = tnc_counts.entry(ctx).or_insert((0.0, 0.0));
        entry.0 += total_alt_snps as f64;
        entry.1 += total_ref_snps as f64;
    }

    let tnc_error_rates = tnc_counts
        .into_iter()
        .map(|(ctx, (alt, ref_count))| {
            let total = alt + ref_count;
            let er = if total > 0.0 {
                let af = alt / total;
                if af > 0.0 && af < 1.0 { af } else { error_rate }
            } else {
                error_rate
            };
            (ctx, er)
        })
        .collect();

    Ok(tnc_error_rates)
}

// ---------------------------------------------------------------------------
// Variant-calling pipeline
// ---------------------------------------------------------------------------

/// Values shared by SNP and indel calling at a single position.
struct SiteContext<'a> {
    contig: &'a str,
    /// 0-based position.
    pos: u32,
    tnc: TrinucleotideContext,
    tnc_er: f64,
    avg: SiteAverages,
    large_entropy: f64,
    small_entropy: f64,
}

/// Candidates and summary statistics for one allele class (SNPs or indels) at a position.
struct ClassCalls {
    candidates: HashSet<BaseCall>,
    counts: HashMap<BaseCall, usize>,
    fwd_counts: HashMap<BaseCall, usize>,
    rev_counts: HashMap<BaseCall, usize>,
    /// Sum of `counts`.
    depth: u64,
    probability: f64,
    fwd_probability: f64,
    rev_probability: f64,
    directive: CallingDirective,
}

/// Emission rules that differ between the SNP and indel classes.
struct ClassRules {
    /// Depth reported in the VCF and used for genotyping.
    reported_depth: u64,
    genotype_error_rate: f64,
    /// Minimum alternate observations required to emit a candidate.
    min_alt_obs: usize,
    /// Use this instead of the directive derived from the pileup.
    directive_override: Option<CallingDirective>,
}

/// Choose candidates, counts and strand-bias statistics for one allele class
/// according to the calling directive.
fn select_class_calls(
    site: &SiteContext,
    fwd: &HashMap<BaseCall, usize>,
    rev: &HashMap<BaseCall, usize>,
    total: &HashMap<BaseCall, usize>,
) -> ClassCalls {
    let (fwd_cands, fwd_probs) = get_count_vec_candidates(fwd, site.tnc_er);
    let (rev_cands, rev_probs) = get_count_vec_candidates(rev, site.tnc_er);
    let (_, total_probs) = get_count_vec_candidates(total, site.tnc_er);

    let directive = find_where_to_call_variants(
        site.tnc.ref_base as char,
        &fwd_cands,
        site.tnc.upstream_base as char,
        site.tnc.downstream_base as char,
    );

    let fwd_prob_sum = fwd_probs.iter().sum::<f64>();
    let rev_prob_sum = rev_probs.iter().sum::<f64>();
    let combined = (fwd_prob_sum + rev_prob_sum).max(1e-10);

    let (candidates, counts, probs) = match &directive {
        CallingDirective::ReferenceSiteOb | CallingDirective::DenovoSiteOb => {
            (rev_cands, rev.clone(), rev_probs)
        }
        CallingDirective::ReferenceSiteOt | CallingDirective::DenovoSiteOt => {
            (fwd_cands, fwd.clone(), fwd_probs)
        }
        CallingDirective::BothStrands | CallingDirective::Indel => {
            let intersection: HashSet<BaseCall> =
                fwd_cands.intersection(&rev_cands).cloned().collect();
            (intersection, total.clone(), total_probs)
        }
    };

    ClassCalls {
        depth: counts.values().sum::<usize>() as u64,
        candidates,
        counts,
        fwd_counts: fwd.clone(),
        rev_counts: rev.clone(),
        probability: probs.iter().sum::<f64>(),
        fwd_probability: fwd_prob_sum / combined,
        rev_probability: rev_prob_sum / combined,
        directive,
    }
}

/// Build, score and (if it passes the ML threshold) emit a variant for each candidate in `class`.
fn emit_variants(
    out: &mut Vec<Variant>,
    site: &SiteContext,
    class: ClassCalls,
    rules: &ClassRules,
    min_depth: u32,
    ml_threshold: f64,
    model_config: &ModelInferenceConfig,
    stranded_read: &ReadNumber,
) {
    if class.candidates.is_empty() || class.depth < min_depth as u64 {
        return;
    }

    let directive = rules
        .directive_override
        .clone()
        .unwrap_or_else(|| class.directive.clone());

    for candidate in &class.candidates {
        let alt_counts = *class.counts.get(candidate).unwrap_or(&0);
        if alt_counts < rules.min_alt_obs {
            continue;
        }

        let ref_call = BaseCall {
            base: candidate.ref_base,
            ref_base: candidate.ref_base,
            deleted_bases: Vec::new(),
            insertion_bases: Vec::new(),
        };
        let alt_forward_count = *class.fwd_counts.get(candidate).unwrap_or(&0) as u32;
        let alt_reverse_count = *class.rev_counts.get(candidate).unwrap_or(&0) as u32;
        let ref_forward_count = *class.fwd_counts.get(&ref_call).unwrap_or(&0) as u32;
        let ref_reverse_count = *class.rev_counts.get(&ref_call).unwrap_or(&0) as u32;

        let (alt_forward_count_r1, alt_forward_count_r2, alt_reverse_count_r1, alt_reverse_count_r2,
            ref_forward_count_r1, ref_forward_count_r2, ref_reverse_count_r1, ref_reverse_count_r2) =
            match stranded_read {
                ReadNumber::R1 => (
                    alt_forward_count,
                    0,
                    alt_reverse_count,
                    0,
                    ref_forward_count,
                    0,
                    ref_reverse_count,
                    0,
                ),
                ReadNumber::R2 => (
                    0,
                    alt_forward_count,
                    0,
                    alt_reverse_count,
                    0,
                    ref_forward_count,
                    0,
                    ref_reverse_count,
                ),
            };

        let mut variant = Variant {
            contig: site.contig.to_string(),
            pos: site.pos + 1,
            reference: candidate.get_reference_allele(),
            alt: candidate.get_alternate_allele(),
            genotype: String::new(),
            score: 0.0,
            depth: rules.reported_depth as u32,
            alt_counts: alt_counts as u32,
            calling_directive: directive.clone(),
            error_rate: site.tnc_er,
            tnc: site.tnc.clone(),
            probability: class.probability,
            average_ref_mapq: site.avg.ref_mapq,
            average_alt_mapq: site.avg.alt_mapq,
            average_ref_mapq_r1: site.avg.ref_mapq_r1,
            average_ref_mapq_r2: site.avg.ref_mapq_r2,
            average_alt_mapq_r1: site.avg.alt_mapq_r1,
            average_alt_mapq_r2: site.avg.alt_mapq_r2,
            average_ref_bq: site.avg.ref_bq,
            average_alt_bq: site.avg.alt_bq,
            average_ref_bq_r1: site.avg.ref_bq_r1,
            average_ref_bq_r2: site.avg.ref_bq_r2,
            average_alt_bq_r1: site.avg.alt_bq_r1,
            average_alt_bq_r2: site.avg.alt_bq_r2,
            avg_ref_dist_from_read_end: site.avg.ref_dist,
            avg_alt_dist_from_read_end: site.avg.alt_dist,
            avg_ref_dist_from_read_end_r1: site.avg.ref_dist_r1,
            avg_ref_dist_from_read_end_r2: site.avg.ref_dist_r2,
            avg_alt_dist_from_read_end_r1: site.avg.alt_dist_r1,
            avg_alt_dist_from_read_end_r2: site.avg.alt_dist_r2,
            avg_ref_insert_size: site.avg.ref_ins,
            avg_alt_insert_size: site.avg.alt_ins,
            fwd_probability: class.fwd_probability,
            rev_probability: class.rev_probability,
            large_local_entropy: site.large_entropy,
            small_local_entropy: site.small_entropy,
            avg_mismatch_per_read: site.avg.mismatch,
            avg_mismatch_per_read_r1: site.avg.mismatch_r1,
            avg_mismatch_per_read_r2: site.avg.mismatch_r2,
            avg_read_length: site.avg.read_length,
            avg_read_length_r1: site.avg.read_length_r1,
            avg_read_length_r2: site.avg.read_length_r2,
            alt_forward_count,
            alt_forward_count_r1,
            alt_forward_count_r2,
            alt_reverse_count,
            alt_reverse_count_r1,
            alt_reverse_count_r2,
            ref_forward_count,
            ref_forward_count_r1,
            ref_forward_count_r2,
            ref_reverse_count,
            ref_reverse_count_r1,
            ref_reverse_count_r2,
            model_probability: 0.0,
            strand_bias: 0.0,
        };

        variant.strand_bias = strand_bias_fisher_pvalue(&variant);
        variant.model_probability = model_probability_score(model_config, &variant);
        if variant.model_probability < ml_threshold {
            continue;
        }

        let genotype = assign_genotype(
            alt_counts,
            rules.reported_depth as usize,
            rules.genotype_error_rate,
        );
        variant.genotype = genotype.genotype;
        variant.score = genotype.score;
        out.push(variant);
    }
}

/// Call all SNP and indel variants in one genome chunk.
///
/// # Arguments
/// * `chunk` - The genome chunk to process
/// * `bam_path` - Path to the BAM file
/// * `ref_seq` - The reference sequence as a byte vector
/// * `min_bq` - Minimum base quality
/// * `min_mapq` - Minimum mapping quality
/// * `min_depth` - Minimum read depth
/// * `end_of_read_cutoff` - End of read cutoff for SNPs
/// * `indel_end_of_read_cutoff` - End of read cutoff for indels
/// * `max_mismatches` - Maximum allowed mismatches in a read
/// * `min_ao` - Minimum alternate allele observations (indels)
/// * `error_rate` - Expected general error rate
///
/// # Returns
/// A vector of Variant instances
fn call_variants_impl(
    chunk: &GenomeChunk,
    bam_path: &str,
    ref_seq: &[u8],
    min_bq: usize,
    min_mapq: usize,
    min_depth: u32,
    end_of_read_cutoff: usize,
    indel_end_of_read_cutoff: usize,
    max_mismatches: u32,
    min_ao: u32,
    error_rate: f64,
    stranded_read: &ReadNumber,
    sequencing_mode: &SequencingMode,
    indel_filter_repeat_limit: usize,
    model_path: &str,
    ml_threshold: f64,
) -> Result<Vec<Variant>, Box<dyn std::error::Error>> {
    let model_config = model_inference_config(model_path);

    let error_map = compute_tnc_error_rates(
        chunk, bam_path, ref_seq, min_bq, min_mapq, min_depth,
        end_of_read_cutoff, indel_end_of_read_cutoff, max_mismatches,
        error_rate, stranded_read, sequencing_mode, indel_filter_repeat_limit,
    )?;

    let mut bam = bam::IndexedReader::from_path(bam_path)?;
    let header = bam.header().to_owned();
    let tid = header.tid(chunk.contig.as_bytes()).ok_or("Contig not found in BAM header")?;
    bam.fetch((tid, chunk.start as i64, chunk.end as i64))?;

    let mut variants = Vec::new();
    let mut pileup_counts = PileupCounts::new();

    let mut fwd_snps   = HashMap::with_capacity(4);
    let mut rev_snps   = HashMap::with_capacity(4);
    let mut fwd_indels = HashMap::with_capacity(4);
    let mut rev_indels = HashMap::with_capacity(4);
    let mut total_snps   = HashMap::with_capacity(4);
    let mut total_indels = HashMap::with_capacity(4);

    for result in bam.pileup() {
        let pileup: Pileup = result?;
        let ref_name = std::str::from_utf8(header.tid2name(pileup.tid()))?;
        let pos = pileup.pos();
        let ref_base = ref_seq[pos as usize];

        let s = compute_pileup_counts(
            &pileup, min_bq, min_mapq, end_of_read_cutoff, indel_end_of_read_cutoff,
            max_mismatches, ref_seq, pos, stranded_read, sequencing_mode, &mut pileup_counts,
            indel_filter_repeat_limit,
        );

        fwd_snps.clear(); rev_snps.clear(); fwd_indels.clear();
        rev_indels.clear(); total_snps.clear(); total_indels.clear();

        distribute_counts(&pileup_counts.fwd,   &mut fwd_snps,   &mut fwd_indels);
        distribute_counts(&pileup_counts.rev,   &mut rev_snps,   &mut rev_indels);
        distribute_counts(&pileup_counts.total, &mut total_snps, &mut total_indels);

        let (upstream, downstream) = flanking_bases(ref_seq, pos as usize);
        let tnc = TrinucleotideContext::new(upstream, ref_base, downstream);
        let tnc_er = error_map.get(&tnc).copied().unwrap_or(error_rate);

        let site = SiteContext {
            contig: ref_name,
            pos,
            tnc,
            tnc_er,
            avg: s.averages(),
            large_entropy: flank_entropy(ref_seq, pos as usize, 50),
            small_entropy: flank_entropy(ref_seq, pos as usize, 15),
        };

        let snps = select_class_calls(&site, &fwd_snps, &rev_snps, &total_snps);
        let indels = select_class_calls(&site, &fwd_indels, &rev_indels, &total_indels);

        let total_depth = snps.depth + indels.depth;
        let total_depth_filtered = total_depth.saturating_sub(s.indel_offset);

        let snp_rules = ClassRules {
            reported_depth: total_depth,
            genotype_error_rate: tnc_er,
            min_alt_obs: 0,
            directive_override: None,
        };
        let indel_rules = ClassRules {
            reported_depth: total_depth_filtered,
            genotype_error_rate: 0.05,
            min_alt_obs: min_ao as usize,
            directive_override: Some(CallingDirective::BothStrands),
        };

        emit_variants(&mut variants, &site, snps, &snp_rules, min_depth, ml_threshold, model_config, stranded_read);
        emit_variants(&mut variants, &site, indels, &indel_rules, min_depth, ml_threshold, model_config, stranded_read);
    }

    Ok(variants)
}

fn call_variants(
    chunk: &GenomeChunk,
    bam_path: &str,
    ref_seq: &[u8],
    min_bq: usize,
    min_mapq: usize,
    min_depth: u32,
    end_of_read_cutoff: usize,
    indel_end_of_read_cutoff: usize,
    max_mismatches: u32,
    min_ao: u32,
    error_rate: f64,
    stranded_read: &ReadNumber,
    indel_filter_repeat_limit: usize,
    model_path: &str,
    ml_threshold: f64,
) -> Result<Vec<Variant>, Box<dyn std::error::Error>> {
    call_variants_impl(
        chunk,
        bam_path,
        ref_seq,
        min_bq,
        min_mapq,
        min_depth,
        end_of_read_cutoff,
        indel_end_of_read_cutoff,
        max_mismatches,
        min_ao,
        error_rate,
        stranded_read,
        &SequencingMode::PairedEnd,
        indel_filter_repeat_limit,
        model_path,
        ml_threshold,
    )
}

// ---------------------------------------------------------------------------
// Application entry point
// ---------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    subscriber_fmt()
        .with_env_filter(EnvFilter::new(args.log_level.as_str()))
        .with_target(false)
        .init();

    workflow(&args)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rust_htslib::faidx;

    #[test]
    #[cfg(feature = "onnx-inference")]
    fn metadata_parser_requires_feature_order() {
        assert_eq!(parse_feature_order_from_metadata("other=value"), None);
        assert_eq!(
            parse_feature_order_from_metadata(r#"["DP","AO"]"#),
            Some(vec!["DP".to_string(), "AO".to_string()])
        );
    }

    #[test]
    #[cfg(feature = "onnx-inference")]
    fn onnx_model_returns_raw_probability_for_fixture() {
        std::env::remove_var("TVC_SKIP_ORT_IN_TESTS");

        let test_ref = "test_assets/chr11.fasta";
        let test_bam = "test_assets/testing_bams/denovo_ot_chr11_134479860_A_G_homo.bam";
        let contig = "chr11";
        let pos = 134479860_u32;

        let ref_reader = faidx::Reader::from_path(test_ref).expect("Failed to open FASTA");
        let seq_len = ref_reader.fetch_seq_len(contig);
        let ref_seq: Vec<u8> = ref_reader
            .fetch_seq(contig, 0, seq_len as usize)
            .expect("Failed to fetch seq")
            .iter()
            .map(|b| b.to_ascii_uppercase())
            .collect();

        let chunk = GenomeChunk::new(contig.to_string(), u64::from(pos), u64::from(pos + 1));
        let variants = call_variants(
            &chunk,
            test_bam,
            &ref_seq,
            20,
            1,
            1,
            5,
            20,
            10,
            1,
            0.005,
            &ReadNumber::R1,
            3,
            "model.onnx",
            0.0,
        )
        .expect("call_variants failed");

        let matching = variants
            .iter()
            .find(|v| v.pos == pos)
            .expect("Expected variant not found");

        let config = model_inference_config("model.onnx");
        let session = load_onnx_session(&config.model_path).expect("Expected ONNX session to load");
        let feature_order_err = resolve_feature_order(&session, config)
            .expect_err("Expected current model metadata to be incompatible with generated features");

        println!(
            "wrapper probability for {}:{} {}>{} = {:.12}; feature-order resolution error = {}",
            matching.contig,
            matching.pos,
            matching.reference,
            matching.alt,
            matching.model_probability,
            feature_order_err,
        );

        assert!(matching.model_probability.is_finite());
        assert!((0.0..=1.0).contains(&matching.model_probability));
        assert_eq!(matching.model_probability, 1.0);
        assert!(
            feature_order_err.contains("unsupported") && feature_order_err.contains("incompatible"),
            "expected feature schema incompatibility, got: {feature_order_err}"
        );
    }

    macro_rules! make_variant_test {
        ($fn_name:ident, $bam_file:expr, $pos:expr, $ref_base:expr, $alt_base:expr, $stranded_read:expr) => {
            #[test]
            fn $fn_name() {
                let test_ref = "test_assets/chr11.fasta";
                let test_bam = concat!("test_assets/testing_bams/", $bam_file);

                let ref_reader = faidx::Reader::from_path(test_ref).expect("Failed to open FASTA");
                let contig = "chr11";
                let seq_len = ref_reader.fetch_seq_len(contig);
                let ref_seq: Vec<u8> = ref_reader
                    .fetch_seq(contig, 0, seq_len as usize)
                    .expect("Failed to fetch seq")
                    .iter()
                    .map(|b| b.to_ascii_uppercase())
                    .collect();

                let chunk = GenomeChunk::new(contig.to_string(), $pos, $pos + 1);
                let variants = call_variants(
                    &chunk, test_bam, &ref_seq,
                    20, 1, 1, 5, 20, 10, 1, 0.005, &$stranded_read, 3, "model.onnx", 0.1
                )
                .expect("call_variants failed");

                if variants.is_empty() {
                    println!("Warning: No variants called");
                }
                for v in &variants {
                    println!("{}", v.to_vcf());
                }

                let matching = variants
                    .iter()
                    .find(|v| v.pos == $pos)
                    .expect("Expected variant not found");

                assert_eq!(matching.contig,    contig,     "Chromosome mismatch");
                assert_eq!(matching.reference, $ref_base,  "REF mismatch");
                assert_eq!(matching.alt,       $alt_base,  "ALT mismatch");
            }
        };
    }

    make_variant_test!(test_both_strands_chr11_8198900_a_c_homo,        "both_strands_chr11_8198900_A_C_homo.bam",        8198900,   "A", "C",      ReadNumber::R1);
    make_variant_test!(test_both_strands_chr11_8198951_t_a_het,         "both_strands_chr11_8198951_T_A_het.bam",         8198951,   "T", "A",      ReadNumber::R1);
    make_variant_test!(test_denovo_ob_chr11_134755809_t_c_homo,         "denovo_ob_chr11_134755809_T_C_homo.bam",         134755809, "T", "C",      ReadNumber::R1);
    make_variant_test!(test_denovo_ob_chr11_134911365_t_c_het,          "denovo_ob_chr11_134911365_T_C_het.bam",          134911365, "T", "C",      ReadNumber::R1);
    make_variant_test!(test_short_hetero_del,                           "chr11:1160400-1160500_short_hetero_del.bam",     1160456,   "AC", "A",     ReadNumber::R1);
    make_variant_test!(test_long_ins_hetero,                            "chr11:228150-228350_long_ins_hetero.bam",        228244,    "C", "CA",     ReadNumber::R1);
    make_variant_test!(test_short_insertion_homo,                       "chr11:6586900-6587100_short_ins_homo.bam",       6586999,   "T", "TG",     ReadNumber::R1);
    make_variant_test!(test_long_ins_homo,                              "chr11:5888900-5889100_long_ins_homo.bam",        5889008,   "C", "CTAGAG", ReadNumber::R1);
    make_variant_test!(test_denovo_ot_chr11_134749303_a_g_het,          "denovo_ot_chr11_134749303_A_G_het.bam",          134749303, "A", "G",      ReadNumber::R1);
    make_variant_test!(test_denovo_ot_chr11_134479860_a_g_homo,         "denovo_ot_chr11_134479860_A_G_homo.bam",         134479860, "A", "G",      ReadNumber::R1);
    make_variant_test!(test_ref_ob_chr11_134012307_c_a_het,             "ref_ob_chr11_134012307_C_A_het.bam",             134012307, "C", "A",      ReadNumber::R1);
    make_variant_test!(test_ref_ob_chr11_134610622_c_t_homo,            "ref_ob_chr11_134610622_C_T_homo.bam",            134610622, "C", "T",      ReadNumber::R1);
    make_variant_test!(test_ref_ot_chr11_134473154_g_a_homo,            "ref_ot_chr11_134473154_G_A_homo.bam",            134473154, "G", "A",      ReadNumber::R1);
    make_variant_test!(test_ref_ot_chr11_8195526_g_a_het,               "ref_ot_chr11_8195526_G_A_het.bam",               8195526,   "G", "A",      ReadNumber::R1);

    #[test]
    fn fisher_exact_strand_bias_is_more_significant_for_extreme_skew() {
        let balanced = fisher_exact_test(2, 2, 2, 2);
        let skewed = fisher_exact_test(4, 0, 0, 4);

        assert!(balanced > 0.0 && balanced <= 1.0);
        assert!(skewed < balanced, "Expected a stronger skew to drive a smaller p-value");
        assert_eq!(fisher_exact_test(0, 0, 0, 0), 1.0);
    }

    #[test]
    fn emitted_variant_carries_computed_strand_bias() {
        let test_ref = "test_assets/chr11.fasta";
        let test_bam = "test_assets/testing_bams/denovo_ot_chr11_134479860_A_G_homo.bam";

        let ref_reader = faidx::Reader::from_path(test_ref).expect("Failed to open FASTA");
        let contig = "chr11";
        let pos = 134479860_u32;
        let seq_len = ref_reader.fetch_seq_len(contig);
        let ref_seq: Vec<u8> = ref_reader
            .fetch_seq(contig, 0, seq_len as usize)
            .expect("Failed to fetch seq")
            .iter()
            .map(|b| b.to_ascii_uppercase())
            .collect();

        let chunk = GenomeChunk::new(contig.to_string(), u64::from(pos), u64::from(pos + 1));
        let variants = call_variants(
            &chunk,
            test_bam,
            &ref_seq,
            20,
            1,
            1,
            5,
            20,
            10,
            1,
            0.005,
            &ReadNumber::R1,
            3,
            "model.onnx",
            0.1,
        )
        .expect("call_variants failed");

        let matching = variants
            .iter()
            .find(|v| v.pos == pos)
            .expect("Expected variant not found");

        assert_eq!(matching.strand_bias, strand_bias_fisher_pvalue(matching));
        assert!(matching.strand_bias > 0.0 && matching.strand_bias < 1.0);
    }

    #[test]
    fn vcf_tnc_uses_alt_base_for_snps_only() {
        let base_variant = Variant {
            contig: "chr1".to_string(),
            pos: 1,
            reference: "T".to_string(),
            alt: "C".to_string(),
            genotype: "0/1".to_string(),
            score: 42.0,
            depth: 10,
            alt_counts: 4,
            calling_directive: CallingDirective::BothStrands,
            error_rate: 0.001,
            tnc: TrinucleotideContext::new(b'A', b'T', b'G'),
            probability: 0.9,
            average_ref_mapq: 30.0,
            average_alt_mapq: 31.0,
            average_ref_mapq_r1: 30.1,
            average_ref_mapq_r2: 29.9,
            average_alt_mapq_r1: 31.1,
            average_alt_mapq_r2: 30.9,
            average_ref_bq: 32.0,
            average_alt_bq: 33.0,
            average_ref_bq_r1: 32.1,
            average_ref_bq_r2: 31.9,
            average_alt_bq_r1: 33.1,
            average_alt_bq_r2: 32.9,
            avg_ref_dist_from_read_end: 5.0,
            avg_alt_dist_from_read_end: 6.0,
            avg_ref_dist_from_read_end_r1: 5.1,
            avg_ref_dist_from_read_end_r2: 4.9,
            avg_alt_dist_from_read_end_r1: 6.1,
            avg_alt_dist_from_read_end_r2: 5.9,
            avg_ref_insert_size: 200.0,
            avg_alt_insert_size: 201.0,
            fwd_probability: 0.8,
            rev_probability: 0.7,
            large_local_entropy: 1.1,
            small_local_entropy: 0.9,
            avg_mismatch_per_read: 0.2,
            avg_mismatch_per_read_r1: 0.21,
            avg_mismatch_per_read_r2: 0.19,
            avg_read_length: 150.0,
            avg_read_length_r1: 151.0,
            avg_read_length_r2: 149.0,
            alt_forward_count: 2,
            alt_forward_count_r1: 1,
            alt_forward_count_r2: 1,
            alt_reverse_count: 2,
            alt_reverse_count_r1: 1,
            alt_reverse_count_r2: 1,
            ref_forward_count: 3,
            ref_forward_count_r1: 2,
            ref_forward_count_r2: 1,
            ref_reverse_count: 3,
            ref_reverse_count_r1: 1,
            ref_reverse_count_r2: 2,
            model_probability: 0.95,
            strand_bias: 0.5,
        };

        let snp_vcf = base_variant.to_vcf();
        assert!(snp_vcf.contains(":A(T>C)G:"), "expected SNP TNC to show ref and ALT: {snp_vcf}");

        let indel_variant = Variant {
            reference: "T".to_string(),
            alt: "TA".to_string(),
            ..base_variant
        };
        let indel_vcf = indel_variant.to_vcf();
        assert!(indel_vcf.contains(":ATG:"), "expected indel TNC to keep REF anchor: {indel_vcf}");
    }

    #[test]
    fn vcf_format_columns_match_values_and_use_alt_strand_counts() {
        let variant = Variant {
            contig: "chr1".to_string(),
            pos: 1,
            reference: "T".to_string(),
            alt: "C".to_string(),
            genotype: "0/1".to_string(),
            score: 42.0,
            depth: 10,
            alt_counts: 4,
            calling_directive: CallingDirective::BothStrands,
            error_rate: 0.001,
            tnc: TrinucleotideContext::new(b'A', b'T', b'G'),
            probability: 0.9,
            average_ref_mapq: 30.0,
            average_alt_mapq: 31.0,
            average_ref_mapq_r1: 30.1,
            average_ref_mapq_r2: 29.9,
            average_alt_mapq_r1: 31.1,
            average_alt_mapq_r2: 30.9,
            average_ref_bq: 32.0,
            average_alt_bq: 33.0,
            average_ref_bq_r1: 32.1,
            average_ref_bq_r2: 31.9,
            average_alt_bq_r1: 33.1,
            average_alt_bq_r2: 32.9,
            avg_ref_dist_from_read_end: 5.0,
            avg_alt_dist_from_read_end: 6.0,
            avg_ref_dist_from_read_end_r1: 5.1,
            avg_ref_dist_from_read_end_r2: 4.9,
            avg_alt_dist_from_read_end_r1: 6.1,
            avg_alt_dist_from_read_end_r2: 5.9,
            avg_ref_insert_size: 200.0,
            avg_alt_insert_size: 201.0,
            fwd_probability: 0.8,
            rev_probability: 0.2,
            large_local_entropy: 1.1,
            small_local_entropy: 0.9,
            avg_mismatch_per_read: 0.2,
            avg_mismatch_per_read_r1: 0.21,
            avg_mismatch_per_read_r2: 0.19,
            avg_read_length: 150.0,
            avg_read_length_r1: 151.0,
            avg_read_length_r2: 149.0,
            alt_forward_count: 2,
            alt_forward_count_r1: 1,
            alt_forward_count_r2: 1,
            alt_reverse_count: 1,
            alt_reverse_count_r1: 1,
            alt_reverse_count_r2: 0,
            ref_forward_count: 7,
            ref_forward_count_r1: 4,
            ref_forward_count_r2: 3,
            ref_reverse_count: 5,
            ref_reverse_count_r1: 2,
            ref_reverse_count_r2: 3,
            model_probability: 0.95,
            strand_bias: 0.5,
        };

        let record = variant.to_vcf();
        let fields: Vec<_> = record.trim_end().split('\t').collect();
        let format_keys: Vec<_> = fields[8].split(':').collect();
        let sample_values: Vec<_> = fields[9].split(':').collect();
        let format_map = format_keys
            .iter()
            .zip(sample_values.iter())
            .map(|(key, value)| (*key, *value))
            .collect::<std::collections::HashMap<_, _>>();

        assert_eq!(format_keys.len(), sample_values.len(), "FORMAT/value column count mismatch: {record}");
        assert!(!format_keys.contains(&"MFC"), "dead FORMAT key should not be emitted: {record}");
        assert_eq!(format_map[&"FWD"], "2", "FWD should be alt-supporting forward reads");
        assert_eq!(format_map[&"REV"], "1", "REV should be alt-supporting reverse reads");
        assert_eq!(format_map[&"TOT"], "3", "TOT should be total alt-supporting reads");
        assert_eq!(format_map[&"FWD_R1"], "1", "FWD_R1 should count alt-supporting forward R1 reads");
        assert_eq!(format_map[&"REV_R2"], "0", "REV_R2 should count alt-supporting reverse R2 reads");
    }

    #[test]
    fn vcf_header_describes_emitted_format_fields() {
        let bam = bam::Reader::from_path(
            "test_assets/testing_bams/denovo_ot_chr11_134749303_A_G_het.bam",
        )
        .expect("Failed to open BAM");
        let header = bam.header().to_owned();
        let vcf_header = get_vcf_header(&header);

        assert!(vcf_header.contains("##FORMAT=<ID=STB,Number=1,Type=Float"));
        assert!(vcf_header.contains("Aggregate right-tail binomial evidence score"));
        assert!(vcf_header.contains("Fraction of strand-separated aggregate evidence contributed by forward reads"));
        assert!(vcf_header.contains("Total alternate-supporting read count across both strands"));
        assert!(vcf_header.contains("##FORMAT=<ID=FWD_R1,Number=1,Type=Float"));
        assert!(vcf_header.contains("##FORMAT=<ID=REDR_R2,Number=1,Type=Float"));
        assert!(!vcf_header.contains("##FORMAT=<ID=MFC,"));
    }

    #[test]
    fn single_end_vcf_omits_mate_specific_format_fields() {
        let variant = Variant {
            contig: "chr1".to_string(),
            pos: 1,
            reference: "T".to_string(),
            alt: "C".to_string(),
            genotype: "0/1".to_string(),
            score: 42.0,
            depth: 10,
            alt_counts: 4,
            calling_directive: CallingDirective::BothStrands,
            error_rate: 0.001,
            tnc: TrinucleotideContext::new(b'A', b'T', b'G'),
            probability: 0.9,
            average_ref_mapq: 30.0,
            average_alt_mapq: 31.0,
            average_ref_mapq_r1: 30.1,
            average_ref_mapq_r2: 29.9,
            average_alt_mapq_r1: 31.1,
            average_alt_mapq_r2: 30.9,
            average_ref_bq: 32.0,
            average_alt_bq: 33.0,
            average_ref_bq_r1: 32.1,
            average_ref_bq_r2: 31.9,
            average_alt_bq_r1: 33.1,
            average_alt_bq_r2: 32.9,
            avg_ref_dist_from_read_end: 5.0,
            avg_alt_dist_from_read_end: 6.0,
            avg_ref_dist_from_read_end_r1: 5.1,
            avg_ref_dist_from_read_end_r2: 4.9,
            avg_alt_dist_from_read_end_r1: 6.1,
            avg_alt_dist_from_read_end_r2: 5.9,
            avg_ref_insert_size: 200.0,
            avg_alt_insert_size: 201.0,
            fwd_probability: 0.8,
            rev_probability: 0.2,
            large_local_entropy: 1.1,
            small_local_entropy: 0.9,
            avg_mismatch_per_read: 0.2,
            avg_mismatch_per_read_r1: 0.21,
            avg_mismatch_per_read_r2: 0.19,
            avg_read_length: 150.0,
            avg_read_length_r1: 151.0,
            avg_read_length_r2: 149.0,
            alt_forward_count: 2,
            alt_forward_count_r1: 2,
            alt_forward_count_r2: 0,
            alt_reverse_count: 1,
            alt_reverse_count_r1: 1,
            alt_reverse_count_r2: 0,
            ref_forward_count: 7,
            ref_forward_count_r1: 7,
            ref_forward_count_r2: 0,
            ref_reverse_count: 5,
            ref_reverse_count_r1: 5,
            ref_reverse_count_r2: 0,
            model_probability: 0.95,
            strand_bias: 0.5,
        };

        let record = variant.to_vcf_for_mode(&SequencingMode::SingleEnd);
        let fields: Vec<_> = record.trim_end().split('\t').collect();
        let format_keys: Vec<_> = fields[8].split(':').collect();

        assert!(!format_keys.iter().any(|key| key.ends_with("_R1") || key.ends_with("_R2")));
        assert!(!record.contains("FWD_R1"));

        let bam = bam::Reader::from_path(
            "test_assets/testing_bams/denovo_ot_chr11_134749303_A_G_het.bam",
        )
        .expect("Failed to open BAM");
        let header = bam.header().to_owned();
        let vcf_header = get_vcf_header_for_mode(&header, &SequencingMode::SingleEnd);

        assert!(!vcf_header.contains("##FORMAT=<ID=FWD_R1,"));
        assert!(!vcf_header.contains("##FORMAT=<ID=AMQR_R2,"));
        assert!(vcf_header.contains("##FORMAT=<ID=FWD,Number=1,Type=Float"));
    }

    fn load_ref_seq(contig: &str) -> Vec<u8> {
        let ref_reader =
            faidx::Reader::from_path("test_assets/chr11.fasta").expect("Failed to open FASTA");
        let seq_len = ref_reader.fetch_seq_len(contig);
        ref_reader
            .fetch_seq(contig, 0, seq_len as usize)
            .expect("Failed to fetch seq")
            .iter()
            .map(|b| b.to_ascii_uppercase())
            .collect()
    }

    #[test]
    fn test_methylation_site_no_variants() {
        let contig = "chr11";
        let ref_seq = load_ref_seq(contig);
        let chunk = GenomeChunk::new(contig.to_string(), 134755601, 134755621);

        let variants = call_variants(
            &chunk,
            "test_assets/testing_bams/methylation_site_chr11_134755601_134755621.bam",
            &ref_seq, 20, 1, 1, 5, 20, 10, 1, 0.005, &ReadNumber::R1, 3, "model.onnx", 0.1
        )
        .expect("call_variants failed");

        let in_range: Vec<_> = variants.iter().filter(|v| v.pos >= 134755601 && v.pos <= 134755621).collect();
        assert!(in_range.is_empty(), "Expected no variants in methylation site BAM");
    }

    #[test]
    fn test_single_ended_reads() {
        let contig = "chr11";
        let ref_seq = load_ref_seq(contig);
        let chunk = GenomeChunk::new(contig.to_string(), 134755601, 134755621);

        let variants = call_variants(
            &chunk,
            "test_assets/testing_bams/methylation_site_chr11_134755601_134755621.single_end.bam",
            &ref_seq, 20, 1, 1, 5, 20, 10, 1, 0.005, &ReadNumber::R1, 3, "model.onnx", 0.1
        )
        .expect("call_variants failed");

        let in_range: Vec<_> = variants.iter().filter(|v| v.pos >= 134755601 && v.pos <= 134755621).collect();
        assert!(in_range.is_empty(), "Expected no variants in single-ended methylation site BAM");
    }

    #[test]
    fn test_read_two_stranded() {
        let contig = "chr11";
        let ref_seq = load_ref_seq(contig);
        let chunk = GenomeChunk::new(contig.to_string(), 134755601, 134755621);

        let variants = call_variants(
            &chunk,
            "test_assets/testing_bams/methylation_site_chr11_134755601_134755621.bam",
            &ref_seq, 20, 1, 1, 5, 20, 10, 1, 0.005, &ReadNumber::R2, 3, "model.onnx", 0.1
        )
        .expect("call_variants failed");

        let in_range: Vec<_> = variants.iter().filter(|v| v.pos >= 134755601 && v.pos <= 134755621).collect();
        assert_eq!(
            in_range.len(), 2,
            "Since R2 was flipped the caller should emit 2 variants, got {}",
            in_range.len()
        );
    }

    #[test]
    fn test_matched_normal_filters_identical_tumor_calls() {
        let contig = "chr11";
        let ref_seq = load_ref_seq(contig);
        let chunk = GenomeChunk::new(contig.to_string(), 134749303, 134749304);
        let bam = "test_assets/testing_bams/denovo_ot_chr11_134749303_A_G_het.bam";

        let tumor_variants = call_variants(
            &chunk,
            bam,
            &ref_seq,
            20,
            1,
            1,
            5,
            20,
            10,
            1,
            0.005,
            &ReadNumber::R1,
            3,
            "model.onnx",
            0.1,
        )
        .expect("tumor call_variants failed");

        assert!(
            !tumor_variants.is_empty(),
            "Expected at least one tumor variant before matched-normal filtering"
        );

        let normal_variants = call_variants(
            &chunk,
            bam,
            &ref_seq,
            20,
            1,
            1,
            5,
            20,
            10,
            1,
            0.005,
            &ReadNumber::R1,
            3,
            "model.onnx",
            0.1,
        )
        .expect("normal call_variants failed");

        let normal_keys: std::collections::HashSet<(String, u32, String, String)> = normal_variants
            .into_iter()
            .map(|v| (v.contig, v.pos, v.reference, v.alt))
            .collect();

        let mut filtered_tumor = tumor_variants;
        filtered_tumor.retain(|v| {
            !normal_keys.contains(&(v.contig.clone(), v.pos, v.reference.clone(), v.alt.clone()))
        });

        assert!(
            filtered_tumor.is_empty(),
            "Expected no tumor variants after matched-normal filtering when using identical BAMs"
        );
    }

    // -----------------------------------------------------------------------
    // Indel filter unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_homopolymer_read_start() {
        let cigar = bam::record::CigarString::from(vec![Cigar::Match(7)]);
        let mut rec = bam::Record::new();
        let seq = b"AAATGCC";
        rec.set(b"r", Some(&cigar), seq, &[255u8; 7]);
        assert!(filter_indels(seq, &rec, 3, 4));

        let cigar2 = bam::record::CigarString::from(vec![Cigar::Match(6)]);
        let mut rec2 = bam::Record::new();
        let seq2 = b"AATGCC";
        rec2.set(b"r", Some(&cigar2), seq2, &[255u8; 6]);
        assert!(!filter_indels(seq2, &rec2, 3, 4));
    }

    #[test]
    fn test_homopolymer_read_end() {
        let cigar = bam::record::CigarString::from(vec![Cigar::Match(6)]);
        let mut rec = bam::Record::new();
        let seq = b"GCCTTT";
        rec.set(b"r", Some(&cigar), seq, &[255u8; 6]);
        assert!(filter_indels(seq, &rec, 3, 4));

        let cigar2 = bam::record::CigarString::from(vec![Cigar::Match(5)]);
        let mut rec2 = bam::Record::new();
        let seq2 = b"GCCTT";
        rec2.set(b"r", Some(&cigar2), seq2, &[255u8; 5]);
        assert!(!filter_indels(seq2, &rec2, 3, 4));
    }

    #[test]
    fn test_dinucleotide_read_start() {
        let cigar = bam::record::CigarString::from(vec![Cigar::Match(6)]);
        let mut rec = bam::Record::new();
        let seq = b"ATATGC";
        rec.set(b"r", Some(&cigar), seq, &[255u8; 6]);
        assert!(filter_indels(seq, &rec, 3, 4));

        let cigar2 = bam::record::CigarString::from(vec![Cigar::Match(6)]);
        let mut rec2 = bam::Record::new();
        let seq2 = b"ATCGTG";
        rec2.set(b"r", Some(&cigar2), seq2, &[255u8; 6]);
        assert!(!filter_indels(seq2, &rec2, 3, 4));
    }

    #[test]
    fn test_dinucleotide_read_end() {
        let cigar = bam::record::CigarString::from(vec![Cigar::Match(6)]);
        let mut rec = bam::Record::new();
        let seq = b"GCCTTT";
        rec.set(b"r", Some(&cigar), seq, &[255u8; 6]);
        assert!(filter_indels(seq, &rec, 3, 4));

        let cigar2 = bam::record::CigarString::from(vec![Cigar::Match(6)]);
        let mut rec2 = bam::Record::new();
        let seq2 = b"GCCTTG";
        rec2.set(b"r", Some(&cigar2), seq2, &[255u8; 6]);
        assert!(!filter_indels(seq2, &rec2, 3, 4));
    }

    #[test]
    fn test_check_soft_clip() {
        let cigar_sc = bam::record::CigarString::from(vec![
            Cigar::SoftClip(5),
            Cigar::Match(10),
            Cigar::SoftClip(3),
        ]);
        let mut rec = bam::Record::new();
        let seq = b"ACGTACGTAC";
        let qual = vec![255u8; 10];
        rec.set(b"r", Some(&cigar_sc), seq, &qual);
        assert!(filter_indels(seq, &rec, 3, 4));

        let cigar_no_sc = bam::record::CigarString::from(vec![Cigar::Match(10)]);
        let mut rec2 = bam::Record::new();
        rec2.set(b"r", Some(&cigar_no_sc), seq, &qual);
        assert!(!filter_indels(seq, &rec2, 3, 4));
    }
}