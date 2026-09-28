import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.index_profile import INDEX_PROFILES, get_index_profile


def test_flat_profile_has_no_training_or_search_parameters():
    resolved = get_index_profile("flat").resolve(document_count=100, top_k=10)

    assert resolved.factory_description == "Flat"
    assert resolved.nlist is None
    assert resolved.training_sample_size is None
    assert resolved.search_parameters == {}


@pytest.mark.parametrize(
    ("profile_key", "factory"),
    [
        ("sq8", "SQ8"),
        ("rabitq-refine-sq8", "RR,RaBitQfs1,Refine(SQ8)"),
    ],
)
def test_small_training_profiles_cap_samples_at_5000(profile_key, factory):
    resolved = get_index_profile(profile_key).resolve(
        document_count=20_000, top_k=10
    )

    assert resolved.factory_description == factory
    assert resolved.training_sample_size == 5_000
    assert resolved.search_parameters == {}


def test_hnsw_profile_resolves_ef_search_and_training_cap():
    resolved = get_index_profile("hnsw32").resolve(document_count=8_000, top_k=10)

    assert resolved.factory_description == "HNSW32"
    assert resolved.training_sample_size == 5_000
    assert resolved.search_parameters == {"hnsw.efSearch": 30}


def test_ivf_profile_resolves_factory_training_and_search_values():
    profile = get_index_profile("ivf-hnsw-rabitq-refine-sq8")
    resolved = profile.resolve(document_count=40_000, top_k=10)

    assert resolved.nlist == 300
    assert resolved.factory_description == (
        "RR,IVF300_HNSW,RaBitQfs1,Refine(SQ8)"
    )
    assert resolved.training_sample_size == 12_000
    assert resolved.search_parameters == {
        "ivf.nprobe": 15,
        "ivf.quantizer.hnsw.efSearch": 30,
        "k_factor": 15,
    }


def test_unknown_profile_lists_available_names():
    with pytest.raises(ValueError, match="flat.*sq8"):
        get_index_profile("missing")


def test_profile_registry_contains_five_named_profiles():
    assert tuple(INDEX_PROFILES) == (
        "flat",
        "sq8",
        "rabitq-refine-sq8",
        "hnsw32",
        "ivf-hnsw-rabitq-refine-sq8",
    )