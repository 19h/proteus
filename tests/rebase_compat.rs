//! Compatibility contracts for the merged metadata and persisted-index layouts.
use proteus::{
    FingerprintConfig, InvertedIndex, Retina, RetinaFormat, RetinaHeader, WordFingerprint,
};
use std::collections::HashMap;

fn fingerprints() -> HashMap<String, WordFingerprint> {
    [
        ("king", vec![1, 2, 3]),
        ("queen", vec![1, 2]),
        ("cat", vec![9]),
    ]
    .into_iter()
    .map(|(w, p)| {
        (
            w.into(),
            WordFingerprint::with_frequency(w.into(), &p, 100, 1.0 / 3.0),
        )
    })
    .collect()
}

#[test]
fn legacy_index_versions_and_new_metadata_layout_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let fps = fingerprints();
    let index = InvertedIndex::from_fingerprints(&fps, 100);
    let expected = Retina::with_index(fps.clone(), 10, FingerprintConfig::default())
        .find_similar_words("king", 10)
        .unwrap();
    let current = dir.path().join("v4.retina");
    RetinaFormat::write_with_index(&current, &fps, 10, &index).unwrap();
    let bytes = std::fs::read(&current).unwrap();
    let current_header = RetinaHeader::from_bytes(&bytes).unwrap();
    assert_eq!(current_header.version, 4);
    // Upstream v3: identical bitmap/string/posting layout, without metadata.
    let delta = current_header.index_offset - 64;
    let mut header = current_header.clone();
    header.version = 3;
    header.index_offset = 64;
    header.fingerprint_offset -= delta;
    header.string_pool_offset -= delta;
    header.inverted_index_offset -= delta;
    let mut v3 = header.to_bytes();
    v3.extend_from_slice(&bytes[current_header.index_offset as usize..]);
    let v3path = dir.path().join("v3.retina");
    std::fs::write(&v3path, &v3).unwrap();
    // Upstream v2: bincode index after the same sorted string pool.
    header.version = 2;
    header.flags = 1;
    let mut v2 = v3[..header.inverted_index_offset as usize].to_vec();
    v2[..64].copy_from_slice(&header.to_bytes());
    v2.extend_from_slice(&bincode::serialize(&index).unwrap());
    let v2path = dir.path().join("v2-index.retina");
    std::fs::write(&v2path, &v2).unwrap();
    // Earlier local v2: metadata, no persisted index.
    let localpath = dir.path().join("v2-metadata.retina");
    RetinaFormat::write(&localpath, &fps, 10).unwrap();
    let mut local = std::fs::read(&localpath).unwrap();
    local[4..6].copy_from_slice(&2u16.to_le_bytes());
    std::fs::write(&localpath, local).unwrap();
    for path in [&current, &v3path, &v2path, &localpath] {
        let retina = Retina::load(path).unwrap();
        assert_eq!(retina.find_similar_words("KING", 10).unwrap(), expected);
        let (_, loaded, persisted) = RetinaFormat::read_with_index(path).unwrap();
        assert_eq!(loaded.len(), 3);
        let mapped = RetinaFormat::mmap(path).unwrap();
        if path != &localpath {
            assert!(persisted.is_some());
            assert_eq!(mapped.find_similar("KING", 10).unwrap(), expected);
            assert_eq!(mapped.find_similar("king", 0).unwrap(), vec![]);
        } else {
            assert!(persisted.is_none());
        }
    }
    let mut bad = bytes;
    let start = current_header.inverted_index_offset as usize;
    bad[start + 8..start + 12].copy_from_slice(&u32::MAX.to_le_bytes());
    let badpath = dir.path().join("bad-posting.retina");
    std::fs::write(&badpath, bad).unwrap();
    assert!(Retina::load(&badpath).is_err());
    assert!(RetinaFormat::mmap(&badpath).is_err());
}

#[test]
fn mmap_cosine_does_not_prune_by_raw_overlap() {
    let mut fps = HashMap::new();
    fps.insert(
        "query".into(),
        WordFingerprint::new("query".into(), &[0, 1, 2], 100),
    );
    fps.insert(
        "perfect".into(),
        WordFingerprint::new("perfect".into(), &[0, 1], 100),
    );
    for i in 0..20 {
        let word = format!("distractor{i}");
        fps.insert(
            word.clone(),
            WordFingerprint::new(word, &(0..50).collect::<Vec<_>>(), 100),
        );
    }
    let retina = Retina::with_index(fps, 10, FingerprintConfig::default());
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index.retina");
    retina.save(&path).unwrap();
    let mapped = RetinaFormat::mmap(&path).unwrap();
    for k in [0, 1, 3, 100, usize::MAX] {
        assert_eq!(
            mapped.find_similar("query", k).unwrap(),
            retina.find_similar_words("query", k).unwrap()
        );
    }
    assert_eq!(mapped.find_similar("query", 1).unwrap()[0].0, "perfect");
}
