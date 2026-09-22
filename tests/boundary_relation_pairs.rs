use std::{
    env,
    fs::File,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use gliner2_rs::boundary::relation_pairs::{
    RelationPair, RelationProposalConfig, RelationTypeSpec, generate_relation_pairs,
};
use ndarray::{Array1, Array2, Array3, Array4, Axis, array, s};
use ndarray_npy::NpzReader;
use serde::Deserialize;

fn spec(heads: &[i64], tails: &[i64], allow_self: bool) -> RelationTypeSpec {
    RelationTypeSpec {
        relation_type: "related".to_owned(),
        head_query_ids: heads.to_vec(),
        tail_query_ids: tails.to_vec(),
        allow_self,
    }
}

fn config(heads: usize, tails: usize, pair_cap: usize, threshold: f32) -> RelationProposalConfig {
    RelationProposalConfig {
        heads_per_relation: heads,
        tails_per_relation: tails,
        pair_cap,
        argument_threshold: threshold,
    }
}

fn discrete(pair: &RelationPair) -> (usize, usize, usize, [usize; 2], [usize; 2]) {
    (
        pair.relation_index,
        pair.head_query_id,
        pair.tail_query_id,
        pair.head,
        pair.tail,
    )
}

#[test]
fn checkpoint_defaults_are_explicit() {
    assert_eq!(RelationProposalConfig::default(), config(32, 32, 64, 0.2));
}

#[test]
fn stable_endpoint_ties_use_coordinates_then_original_flat_index() -> Result<()> {
    let indices = array![
        [[5, 6], [1, 2], [1, 2], [3, 4]],
        [[3, 4], [1, 3], [8, 9], [0, 1]],
        [[10, 11], [7, 8], [7, 8], [9, 10]],
    ];
    let valid = Array2::from_elem((3, 4), true);
    let query_mask = array![true, true, true];
    let logits = Array2::zeros((3, 4));
    let pairs = generate_relation_pairs(
        indices.view(),
        valid.view(),
        query_mask.view(),
        logits.view(),
        &[spec(&[0, 1], &[2], false)],
        config(6, 1, 6, 0.5),
    )?;

    let expected = [
        (0, 1, 2, [0, 1], [7, 8]),
        (0, 0, 2, [1, 2], [7, 8]),
        (0, 0, 2, [1, 2], [7, 8]),
        (0, 1, 2, [1, 3], [7, 8]),
        (0, 0, 2, [3, 4], [7, 8]),
        (0, 1, 2, [3, 4], [7, 8]),
    ];
    assert_eq!(pairs.iter().map(discrete).collect::<Vec<_>>(), expected);
    assert!(
        pairs
            .iter()
            .all(|pair| { pair.head_probability == 0.5 && pair.tail_probability == 0.5 })
    );
    Ok(())
}

#[test]
fn inclusive_threshold_uses_raw_untempered_sigmoid() -> Result<()> {
    let indices = array![[[0, 1], [2, 3]], [[4, 5], [6, 7]]];
    let valid = Array2::from_elem((2, 2), true);
    let pairs = generate_relation_pairs(
        indices.view(),
        valid.view(),
        array![true, true].view(),
        array![[0.0, -0.0001], [0.0, -0.0001]].view(),
        &[spec(&[0], &[1], false)],
        config(8, 8, 8, 0.5),
    )?;
    assert_eq!(pairs.len(), 1);
    assert_eq!(discrete(&pairs[0]), (0, 0, 1, [0, 1], [4, 5]));
    assert_eq!(pairs[0].head_probability, 0.5);
    assert_eq!(pairs[0].tail_probability, 0.5);
    Ok(())
}

#[test]
fn source_sigmoid_rounding_controls_endpoint_cap() -> Result<()> {
    let pairs = generate_relation_pairs(
        array![[[0, 1], [10, 11]], [[20, 21], [-1, -1]]].view(),
        array![[true, true], [true, false]].view(),
        array![true, true].view(),
        array![[-1.3859999_f32, -1.3859998_f32], [0.0, 0.0]].view(),
        &[spec(&[0], &[1], false)],
        config(1, 1, 1, 0.2),
    )?;
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0].head, [0, 1]);
    assert_eq!(pairs[0].head_probability.to_bits(), 0x3e4c_d927);
    Ok(())
}

#[test]
fn source_sigmoid_rounding_controls_product_rank() -> Result<()> {
    let pairs = generate_relation_pairs(
        array![[[0, 1], [10, 11]], [[20, 21], [-1, -1]]].view(),
        array![[true, true], [true, false]].view(),
        array![true, true].view(),
        array![[-1.3859999_f32, -1.3859998_f32], [0.0, 0.0]].view(),
        &[spec(&[0], &[1], false)],
        config(2, 1, 1, 0.2),
    )?;
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0].head, [0, 1]);
    assert_eq!(pairs[0].head_probability.to_bits(), 0x3e4c_d927);
    Ok(())
}

#[test]
fn saturated_sigmoid_ties_do_not_fall_back_to_raw_logits() -> Result<()> {
    let pairs = generate_relation_pairs(
        array![[[9, 10], [1, 2]], [[20, 21], [-1, -1]]].view(),
        array![[true, true], [true, false]].view(),
        array![true, true].view(),
        array![[100.0, 90.0], [80.0, 70.0]].view(),
        &[spec(&[0], &[1], false)],
        config(2, 1, 2, 0.2),
    )?;
    assert_eq!(pairs.len(), 2);
    assert_eq!(pairs[0].head, [1, 2]);
    assert_eq!(pairs[1].head, [9, 10]);
    assert_eq!(pairs[0].head_probability, 1.0);
    assert_eq!(pairs[1].head_probability, 1.0);
    Ok(())
}

#[test]
fn masks_ignore_garbage_and_invalid_query_ids_are_ignored() -> Result<()> {
    let indices = array![
        [[0, 1], [-99, -7], [2, 3]],
        [[4, 5], [6, 7], [8, 9]],
        [[-8, -9], [-7, -10], [-1, -1]],
    ];
    let valid = array![
        [true, false, false],
        [true, false, false],
        [true, true, true]
    ];
    let pairs = generate_relation_pairs(
        indices.view(),
        valid.view(),
        array![true, true, false].view(),
        array![[1.0, 100.0, 100.0], [1.0, 100.0, 100.0], [100.0; 3]].view(),
        &[spec(&[-1, 0, 99], &[1, 200], false)],
        RelationProposalConfig::default(),
    )?;
    assert_eq!(pairs.len(), 1);
    assert_eq!(discrete(&pairs[0]), (0, 0, 1, [0, 1], [4, 5]));
    Ok(())
}

#[test]
fn exact_span_self_policy_and_relation_major_order_are_preserved() -> Result<()> {
    let indices = array![[[0, 1]], [[0, 1]], [[2, 3]]];
    let valid = Array2::from_elem((3, 1), true);
    let query_mask = array![true, true, true];
    let logits = Array2::zeros((3, 1));
    let specs = [
        spec(&[0], &[1], false),
        spec(&[0], &[1], true),
        spec(&[0, 1], &[2], false),
    ];
    let pairs = generate_relation_pairs(
        indices.view(),
        valid.view(),
        query_mask.view(),
        logits.view(),
        &specs,
        config(4, 4, 4, 0.0),
    )?;
    assert_eq!(
        pairs
            .iter()
            .map(|pair| pair.relation_index)
            .collect::<Vec<_>>(),
        [1, 2, 2]
    );
    assert_eq!(pairs[0].head, [0, 1]);
    assert_eq!(pairs[0].tail, [0, 1]);
    assert_eq!((pairs[1].head_query_id, pairs[2].head_query_id), (0, 1));
    Ok(())
}

#[test]
fn endpoint_and_pair_caps_keep_stable_head_major_cartesian_ties() -> Result<()> {
    let indices = array![[[0, 1], [2, 3], [4, 5]], [[10, 11], [12, 13], [14, 15]],];
    let pairs = generate_relation_pairs(
        indices.view(),
        Array2::from_elem((2, 3), true).view(),
        array![true, true].view(),
        array![[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]].view(),
        &[spec(&[0], &[1], false)],
        config(2, 2, 3, 0.0),
    )?;
    assert_eq!(
        pairs.iter().map(discrete).collect::<Vec<_>>(),
        [
            (0, 0, 1, [0, 1], [10, 11]),
            (0, 0, 1, [0, 1], [12, 13]),
            (0, 0, 1, [2, 3], [10, 11]),
        ]
    );
    Ok(())
}

#[test]
fn broadcast_and_non_contiguous_views_match_without_forcing_copies() -> Result<()> {
    let shared_indices = array![[[2, 3], [0, 1]]];
    let shared_valid = array![[true, true]];
    let shared_logits = array![[0.0, 0.0]];
    let broadcast_indices = shared_indices
        .broadcast((2, 2, 2))
        .context("indices broadcast")?;
    let broadcast_valid = shared_valid.broadcast((2, 2)).context("valid broadcast")?;
    let broadcast_logits = shared_logits.broadcast((2, 2)).context("logit broadcast")?;
    let expected = generate_relation_pairs(
        broadcast_indices,
        broadcast_valid,
        array![true, true].view(),
        broadcast_logits,
        &[spec(&[0], &[1], false)],
        config(2, 2, 4, 0.5),
    )?;

    let storage_indices = array![
        [[2, 3], [99, 100], [0, 1], [101, 102]],
        [[2, 3], [99, 100], [0, 1], [101, 102]],
    ];
    let storage_valid = Array2::from_elem((2, 4), true);
    let storage_logits = Array2::zeros((2, 4));
    let actual = generate_relation_pairs(
        storage_indices.slice(s![.., ..;2, ..]),
        storage_valid.slice(s![.., ..;2]),
        array![true, true].view(),
        storage_logits.slice(s![.., ..;2]),
        &[spec(&[0], &[1], false)],
        config(2, 2, 4, 0.5),
    )?;
    assert_eq!(actual, expected);
    assert_eq!(
        actual.iter().map(discrete).collect::<Vec<_>>(),
        [(0, 0, 1, [0, 1], [2, 3]), (0, 0, 1, [2, 3], [0, 1]),]
    );
    Ok(())
}

#[test]
fn pytorch_vector_blocks_match_realistic_pool_sizes_and_strided_scalar_path() -> Result<()> {
    let rounding_probe = f32::from_bits(0xbfb1_a843);
    for query_count in [2, 4] {
        let mut indices = Array3::from_elem((query_count, 192, 2), -1_i64);
        indices[(0, 0, 0)] = 0;
        indices[(0, 0, 1)] = 1;
        indices[(1, 0, 0)] = 2;
        indices[(1, 0, 1)] = 3;
        let mut valid = Array2::from_elem((query_count, 192), false);
        valid[(0, 0)] = true;
        valid[(1, 0)] = true;
        let mut logits = Array2::from_elem((query_count, 192), rounding_probe);
        logits[(1, 0)] = 0.0;
        let query_mask = Array1::from_elem(query_count, true);
        let pairs = generate_relation_pairs(
            indices.view(),
            valid.view(),
            query_mask.view(),
            logits.view(),
            &[spec(&[0], &[1], false)],
            config(1, 1, 1, 0.19),
        )?;
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].head_probability.to_bits(), 0x3e4c_8780);
    }

    let mut indices = Array3::from_elem((2, 384, 2), -1_i64);
    indices[(0, 0, 0)] = 0;
    indices[(0, 0, 1)] = 1;
    indices[(1, 0, 0)] = 2;
    indices[(1, 0, 1)] = 3;
    let mut valid = Array2::from_elem((2, 384), false);
    valid[(0, 0)] = true;
    valid[(1, 0)] = true;
    let mut logits = Array2::from_elem((2, 384), rounding_probe);
    logits[(1, 0)] = 0.0;
    let pairs = generate_relation_pairs(
        indices.slice(s![.., ..;2, ..]),
        valid.slice(s![.., ..;2]),
        array![true, true].view(),
        logits.slice(s![.., ..;2]),
        &[spec(&[0], &[1], false)],
        config(1, 1, 1, 0.19),
    )?;
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0].head_probability.to_bits(), 0x3e4c_877f);

    let shared_logits = array![[rounding_probe]];
    let broadcast_logits = shared_logits
        .broadcast((2, 192))
        .context("rounding probe broadcast")?;
    let pairs = generate_relation_pairs(
        indices.slice(s![.., ..;2, ..]),
        valid.slice(s![.., ..;2]),
        array![true, true].view(),
        broadcast_logits,
        &[spec(&[0], &[1], false)],
        config(1, 1, 1, 0.19),
    )?;
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0].head_probability.to_bits(), 0x3e4c_8780);

    let transposed_storage = Array2::from_elem((192, 2), rounding_probe);
    let pairs = generate_relation_pairs(
        indices.slice(s![.., ..;2, ..]),
        valid.slice(s![.., ..;2]),
        array![true, true].view(),
        transposed_storage.view().reversed_axes(),
        &[spec(&[0], &[1], false)],
        config(1, 1, 1, 0.19),
    )?;
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0].head_probability.to_bits(), 0x3e4c_8780);

    let mut short_indices = Array3::from_elem((2, 5, 2), -1_i64);
    let mut short_valid = Array2::from_elem((2, 5), false);
    for candidate in 0..5 {
        short_indices[(0, candidate, 0)] = i64::try_from(candidate * 2)?;
        short_indices[(0, candidate, 1)] = i64::try_from(candidate * 2 + 1)?;
        short_valid[(0, candidate)] = true;
    }
    short_indices[(1, 0, 0)] = 20;
    short_indices[(1, 0, 1)] = 21;
    short_valid[(1, 0)] = true;
    let short_storage = Array2::from_elem((5, 2), rounding_probe);
    let pairs = generate_relation_pairs(
        short_indices.view(),
        short_valid.view(),
        array![true, true].view(),
        short_storage.view().reversed_axes(),
        &[spec(&[0], &[1], false)],
        config(5, 1, 5, 0.19),
    )?;
    assert_eq!(pairs.len(), 5);
    assert!(
        pairs[..4]
            .iter()
            .all(|pair| pair.head_probability.to_bits() == 0x3e4c_8780)
    );
    assert_eq!(pairs[4].head_probability.to_bits(), 0x3e4c_877f);
    Ok(())
}

#[test]
fn pytorch_column_gaps_and_broadcast_tails_match() -> Result<()> {
    let rounding_probe = f32::from_bits(0xbfb1_a843);
    let mut indices = Array3::from_elem((9, 2, 2), -1_i64);
    let mut valid = Array2::from_elem((9, 2), false);
    for query in 0..9 {
        indices[(query, 0, 0)] = i64::try_from(query * 2)?;
        indices[(query, 0, 1)] = i64::try_from(query * 2 + 1)?;
        valid[(query, 0)] = true;
    }
    let storage = Array2::from_elem((2, 18), rounding_probe);
    let logits = storage.slice(s![.., ..9]).reversed_axes();
    let pairs = generate_relation_pairs(
        indices.view(),
        valid.view(),
        Array1::from_elem(9, true).view(),
        logits,
        &[spec(&(0_i64..9).collect::<Vec<_>>(), &[0], true)],
        config(9, 1, 9, 0.19),
    )?;
    assert_eq!(pairs.len(), 9);
    assert!(
        pairs[..8]
            .iter()
            .all(|pair| pair.head_probability.to_bits() == 0x3e4c_8780)
    );
    assert_eq!(pairs[8].head_probability.to_bits(), 0x3e4c_877f);

    let indices = array![
        [[0, 1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
        [[2, 3], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]
    ];
    let valid = array![
        [true, false, false, false, false],
        [true, false, false, false, false]
    ];
    let shared = array![[rounding_probe]];
    let row = Array2::from_elem((1, 5), rounding_probe);
    let column = Array2::from_elem((2, 1), rounding_probe);
    for (name, logits, expected_bits) in [
        (
            "scalar broadcast",
            shared.broadcast((2, 5)).context("scalar broadcast")?,
            0x3e4c_8780,
        ),
        (
            "row broadcast",
            row.broadcast((2, 5)).context("row broadcast")?,
            0x3e4c_877f,
        ),
        (
            "column broadcast",
            column.broadcast((2, 5)).context("column broadcast")?,
            0x3e4c_877f,
        ),
    ] {
        let pairs = generate_relation_pairs(
            indices.view(),
            valid.view(),
            array![true, true].view(),
            logits,
            &[spec(&[0], &[1], false)],
            config(1, 1, 1, 0.19),
        )?;
        assert_eq!(pairs.len(), 1, "{name}");
        assert_eq!(pairs[0].head_probability.to_bits(), expected_bits, "{name}");
        assert_eq!(pairs[0].tail_probability.to_bits(), expected_bits, "{name}");
    }
    Ok(())
}

#[test]
fn empty_relations_bypass_axes_but_nonempty_inputs_are_strictly_validated() {
    let empty_indices = Array3::<i64>::zeros((0, 0, 2));
    let empty_mask = Array2::<bool>::from_elem((0, 0), false);
    let empty_queries = Array1::<bool>::from_elem(0, false);
    let empty_logits = Array2::<f32>::zeros((0, 0));
    assert!(
        generate_relation_pairs(
            empty_indices.view(),
            empty_mask.view(),
            empty_queries.view(),
            empty_logits.view(),
            &[],
            RelationProposalConfig::default(),
        )
        .unwrap()
        .is_empty()
    );
    assert!(
        generate_relation_pairs(
            empty_indices.view(),
            empty_mask.view(),
            empty_queries.view(),
            empty_logits.view(),
            &[spec(&[0], &[0], false)],
            RelationProposalConfig::default(),
        )
        .is_err()
    );

    let no_candidates = Array3::<i64>::zeros((1, 0, 2));
    assert!(
        generate_relation_pairs(
            no_candidates.view(),
            Array2::<bool>::from_elem((1, 0), false).view(),
            array![true].view(),
            Array2::<f32>::zeros((1, 0)).view(),
            &[spec(&[0], &[0], false)],
            RelationProposalConfig::default(),
        )
        .is_err()
    );
}

#[test]
fn malformed_shapes_values_caps_and_live_spans_fail_closed() {
    let indices = array![[[0, 1]]];
    let valid = array![[true]];
    let queries = array![true];
    let logits = array![[0.0]];
    let specs = [spec(&[0], &[0], true)];

    assert!(
        generate_relation_pairs(
            Array3::<i64>::zeros((1, 1, 3)).view(),
            valid.view(),
            queries.view(),
            logits.view(),
            &specs,
            RelationProposalConfig::default()
        )
        .is_err()
    );
    assert!(
        generate_relation_pairs(
            indices.view(),
            Array2::from_elem((1, 2), true).view(),
            queries.view(),
            logits.view(),
            &specs,
            RelationProposalConfig::default()
        )
        .is_err()
    );
    assert!(
        generate_relation_pairs(
            indices.view(),
            valid.view(),
            queries.view(),
            array![[f32::NAN]].view(),
            &specs,
            RelationProposalConfig::default()
        )
        .is_err()
    );
    assert!(
        generate_relation_pairs(
            array![[[2, 2]]].view(),
            valid.view(),
            queries.view(),
            logits.view(),
            &specs,
            RelationProposalConfig::default()
        )
        .is_err()
    );
    assert!(
        generate_relation_pairs(
            array![[[-1, 2]]].view(),
            valid.view(),
            queries.view(),
            logits.view(),
            &specs,
            RelationProposalConfig::default()
        )
        .is_err()
    );

    let reverse_storage = array![[0.0, 1.0]];
    let error = generate_relation_pairs(
        array![[[0, 1], [2, 3]]].view(),
        array![[true, true]].view(),
        queries.view(),
        reverse_storage.slice(s![.., ..;-1]),
        &specs,
        RelationProposalConfig::default(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("negative-stride pair_logits"));

    for bad_config in [
        config(0, 1, 1, 0.2),
        config(1, 0, 1, 0.2),
        config(1, 1, 0, 0.2),
        config(1, 1, 1, f32::NAN),
        config(1, 1, 1, -0.1),
        config(usize::MAX, 2, 1, 0.2),
    ] {
        assert!(
            generate_relation_pairs(
                indices.view(),
                valid.view(),
                queries.view(),
                logits.view(),
                &specs,
                bad_config
            )
            .is_err()
        );
    }
}

const GOLDEN_IDS: [&str; 4] = [
    "relation_employment",
    "relation_founded",
    "relation_location",
    "relation_multiple_types",
];

fn full_fixture_root() -> PathBuf {
    env::var_os("GLINER2_TEST_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")))
        .join("fixtures/gliner2.5-base-v1")
}

fn committed_subset_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/gliner2.5-base-v1-subset")
}

fn assert_close(case: &str, field: &str, index: usize, actual: f32, expected: f32) {
    assert!(actual.is_finite(), "{case}/{field}[{index}] is not finite");
    assert!(
        (actual - expected).abs() <= 1e-4 + 1e-3 * expected.abs(),
        "{case}/{field}[{index}]: {actual} != {expected}"
    );
}

fn run_golden(root: &Path, case: &str, specs: &[RelationTypeSpec]) -> Result<()> {
    let path = root.join(format!("{case}.npz"));
    let mut npz = NpzReader::new(File::open(&path).with_context(|| path.display().to_string())?)?;
    let indices: Array4<i64> = npz.by_name("boundary_head_0_indices")?;
    let valid: Array3<bool> = npz.by_name("boundary_head_0_valid_mask")?;
    let query_mask: Array2<bool> = npz.by_name("boundary_head_0_query_mask")?;
    let logits: Array3<f32> = npz.by_name("boundary_head_0_pair_logits")?;
    ensure!(
        indices.len_of(Axis(0)) == 1,
        "{case}: indices batch is not one"
    );
    ensure!(valid.len_of(Axis(0)) == 1, "{case}: mask batch is not one");
    ensure!(
        query_mask.nrows() == 1,
        "{case}: query mask batch is not one"
    );
    ensure!(
        logits.len_of(Axis(0)) == 1,
        "{case}: logits batch is not one"
    );

    let actual = generate_relation_pairs(
        indices.index_axis(Axis(0), 0),
        valid.index_axis(Axis(0), 0),
        query_mask.index_axis(Axis(0), 0),
        logits.index_axis(Axis(0), 0),
        specs,
        RelationProposalConfig::default(),
    )?;

    let batch: Array1<i64> = npz.by_name("relation_scorer_0_batch_index")?;
    let relation: Array1<i64> = npz.by_name("relation_scorer_0_relation_index")?;
    let head_start: Array1<i64> = npz.by_name("relation_scorer_0_head_start")?;
    let head_end: Array1<i64> = npz.by_name("relation_scorer_0_head_end")?;
    let tail_start: Array1<i64> = npz.by_name("relation_scorer_0_tail_start")?;
    let tail_end: Array1<i64> = npz.by_name("relation_scorer_0_tail_end")?;
    let head_probability: Array1<f32> = npz.by_name("relation_scorer_0_head_prob")?;
    let tail_probability: Array1<f32> = npz.by_name("relation_scorer_0_tail_prob")?;
    let pair_mask: Array1<bool> = npz.by_name("relation_scorer_0_pair_mask")?;
    ensure!(
        [
            relation.len(),
            head_start.len(),
            head_end.len(),
            tail_start.len(),
            tail_end.len(),
            head_probability.len(),
            tail_probability.len(),
            pair_mask.len(),
        ]
        .into_iter()
        .all(|length| length == batch.len()),
        "{case}: relation golden arrays have inconsistent lengths"
    );
    assert_eq!(actual.len(), batch.len(), "{case}: compact pair count");

    for (index, pair) in actual.iter().enumerate() {
        assert_eq!(batch[index], 0, "{case}: batch_index[{index}]");
        assert!(pair_mask[index], "{case}: pair_mask[{index}] is false");
        let relation_index = usize::try_from(relation[index])?;
        assert_eq!(
            pair.relation_index, relation_index,
            "{case}: relation[{index}]"
        );
        assert_eq!(
            pair.head_query_id,
            relation_index * 2,
            "{case}: head query[{index}]"
        );
        assert_eq!(
            pair.tail_query_id,
            relation_index * 2 + 1,
            "{case}: tail query[{index}]"
        );
        assert_eq!(
            pair.head,
            [
                usize::try_from(head_start[index])?,
                usize::try_from(head_end[index])?
            ],
            "{case}: head span[{index}]"
        );
        assert_eq!(
            pair.tail,
            [
                usize::try_from(tail_start[index])?,
                usize::try_from(tail_end[index])?
            ],
            "{case}: tail span[{index}]"
        );
        assert_close(
            case,
            "head_probability",
            index,
            pair.head_probability,
            head_probability[index],
        );
        assert_close(
            case,
            "tail_probability",
            index,
            pair.tail_probability,
            tail_probability[index],
        );
    }
    Ok(())
}

#[derive(Deserialize)]
struct VectorPayload {
    format_version: u32,
    upstream_commit: String,
    relation_source_sha256: String,
    cases: Vec<VectorCase>,
}

#[derive(Deserialize)]
struct VectorCase {
    name: String,
    source_pool_count: usize,
    indices: Vec<Vec<[i64; 2]>>,
    valid_mask: Vec<Vec<bool>>,
    query_mask: Vec<bool>,
    pair_logits: Vec<Vec<f32>>,
    specs: Vec<VectorSpec>,
    config: VectorConfig,
    expected: Vec<VectorExpected>,
    oracle_pair_mask: Vec<bool>,
}

#[derive(Deserialize)]
struct VectorSpec {
    relation_type: String,
    head_query_ids: Vec<i64>,
    tail_query_ids: Vec<i64>,
    allow_self: bool,
}

#[derive(Deserialize)]
struct VectorConfig {
    heads_per_relation: usize,
    tails_per_relation: usize,
    pair_cap: usize,
    argument_threshold: f32,
}

#[derive(Deserialize)]
struct VectorExpected {
    relation_index: usize,
    head_query_id: usize,
    tail_query_id: usize,
    head: [usize; 2],
    tail: [usize; 2],
    head_probability: f32,
    tail_probability: f32,
}

fn run_relation_pair_vectors(path: &Path) -> Result<()> {
    let payload: VectorPayload = serde_json::from_reader(
        File::open(path)
            .with_context(|| format!("missing strict oracle vectors at {}", path.display()))?,
    )?;
    ensure!(
        payload.format_version == 1,
        "unsupported relation vector format"
    );
    ensure!(
        payload.upstream_commit == "d7c727458bf6929bc9ef5ee04e13c3f717a7c455",
        "relation vectors came from an unexpected upstream commit"
    );
    ensure!(
        payload.relation_source_sha256
            == "59e0b20c040e95a0a0cd0c59fe5d5563f6c85d9d5f060138bda583a7f4a90cbf",
        "relation vectors came from unexpected upstream source"
    );

    for case in payload.cases {
        let query_count = case.indices.len();
        ensure!(query_count > 0, "{} has no queries", case.name);
        let candidate_count = case.indices[0].len();
        ensure!(candidate_count > 0, "{} has no candidates", case.name);
        ensure!(
            case.indices.iter().all(|row| row.len() == candidate_count)
                && case.valid_mask.len() == query_count
                && case
                    .valid_mask
                    .iter()
                    .all(|row| row.len() == candidate_count)
                && case.query_mask.len() == query_count
                && case.pair_logits.len() == query_count
                && case
                    .pair_logits
                    .iter()
                    .all(|row| row.len() == candidate_count),
            "{} has inconsistent source shapes",
            case.name
        );
        ensure!(
            case.source_pool_count == query_count * candidate_count,
            "{} source_pool_count is inconsistent",
            case.name
        );
        ensure!(
            case.oracle_pair_mask.len() == case.expected.len()
                && case.oracle_pair_mask.iter().all(|&live| live),
            "{} compact oracle pair mask is malformed",
            case.name
        );

        let indices = Array3::from_shape_vec(
            (query_count, candidate_count, 2),
            case.indices
                .iter()
                .flatten()
                .flat_map(|span| span.iter().copied())
                .collect(),
        )?;
        let valid = Array2::from_shape_vec(
            (query_count, candidate_count),
            case.valid_mask.iter().flatten().copied().collect(),
        )?;
        let query_mask = Array1::from_vec(case.query_mask);
        let logits = Array2::from_shape_vec(
            (query_count, candidate_count),
            case.pair_logits.iter().flatten().copied().collect(),
        )?;
        let specs = case
            .specs
            .into_iter()
            .map(|item| RelationTypeSpec {
                relation_type: item.relation_type,
                head_query_ids: item.head_query_ids,
                tail_query_ids: item.tail_query_ids,
                allow_self: item.allow_self,
            })
            .collect::<Vec<_>>();
        let actual = generate_relation_pairs(
            indices.view(),
            valid.view(),
            query_mask.view(),
            logits.view(),
            &specs,
            config(
                case.config.heads_per_relation,
                case.config.tails_per_relation,
                case.config.pair_cap,
                case.config.argument_threshold,
            ),
        )?;
        assert_eq!(
            actual.len(),
            case.expected.len(),
            "{} pair count",
            case.name
        );
        for (index, (actual, expected)) in actual.iter().zip(&case.expected).enumerate() {
            assert_eq!(
                discrete(actual),
                (
                    expected.relation_index,
                    expected.head_query_id,
                    expected.tail_query_id,
                    expected.head,
                    expected.tail,
                ),
                "{} discrete pair {index}",
                case.name
            );
            assert_eq!(
                actual.head_probability.to_bits(),
                expected.head_probability.to_bits(),
                "{} head probability {index}",
                case.name
            );
            assert_eq!(
                actual.tail_probability.to_bits(),
                expected.tail_probability.to_bits(),
                "{} tail probability {index}",
                case.name
            );
        }
    }
    Ok(())
}

#[test]
fn committed_relation_employment_golden_always_runs() -> Result<()> {
    run_golden(
        &committed_subset_root(),
        "relation_employment",
        &[spec(&[0], &[1], false)],
    )
}

#[test]
fn strict_full_relation_goldens_and_runtime_vectors_match() -> Result<()> {
    if env::var("GLINER2_BOUNDARY_FULL_FIXTURES").as_deref() != Ok("1") {
        eprintln!("SKIP: additional heavyweight relation fixtures require strict mode");
        return Ok(());
    }

    let root = full_fixture_root();
    let present = GOLDEN_IDS
        .iter()
        .filter(|id| root.join(format!("{id}.npz")).is_file())
        .count();
    ensure!(
        present == GOLDEN_IDS.len(),
        "fixed relation fixture set is incomplete at {}: found {present}/{}",
        root.display(),
        GOLDEN_IDS.len()
    );

    run_golden(&root, GOLDEN_IDS[0], &[spec(&[0], &[1], false)])?;
    run_golden(&root, GOLDEN_IDS[1], &[spec(&[0], &[1], false)])?;
    run_golden(&root, GOLDEN_IDS[2], &[spec(&[0], &[1], false)])?;
    run_golden(
        &root,
        GOLDEN_IDS[3],
        &[spec(&[0], &[1], false), spec(&[2], &[3], false)],
    )?;
    run_relation_pair_vectors(&root.join("relation-aux/relation_pair_vectors.json"))
}
