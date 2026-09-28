"""Declarative FAISS index profiles used by Pocket Dimension retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class ResolvedIndexProfile:
    key: str
    factory_description: str
    nlist: int | None
    training_sample_size: int | None
    search_parameters: dict[str, int]

    def metadata(self, training_sample_count: int) -> dict:
        return {
            "name": self.key,
            "factory": self.factory_description,
            "nlist": self.nlist,
            "training_sample_count": training_sample_count,
            "search_parameters": self.search_parameters,
        }


@dataclass(frozen=True)
class IndexProfile:
    key: str
    factory_template: str
    training_sample_cap: int | None = None
    training_samples_per_list: int | None = None
    nlist_scale: float | None = None
    hnsw_ef_search_per_k: int | None = None
    ivf_nprobe_fraction: float | None = None
    ivf_nprobe_minimum: int | None = None
    quantizer_ef_search_per_nprobe: int | None = None
    k_factor: int | None = None

    def resolve(self, document_count: int, top_k: int) -> ResolvedIndexProfile:
        nlist = None
        if self.nlist_scale is not None:
            nlist = max(1, int(self.nlist_scale * sqrt(document_count)))

        factory_description = self.factory_template.format(nlist=nlist)
        if self.training_sample_cap is not None:
            training_sample_size = min(self.training_sample_cap, document_count)
        elif self.training_samples_per_list is not None and nlist is not None:
            training_sample_size = min(
                self.training_samples_per_list * nlist, document_count
            )
        else:
            training_sample_size = None

        search_parameters = {}
        if self.hnsw_ef_search_per_k is not None:
            search_parameters["hnsw.efSearch"] = self.hnsw_ef_search_per_k * top_k
        if self.ivf_nprobe_fraction is not None and nlist is not None:
            nprobe = max(
                self.ivf_nprobe_minimum or 0,
                int(self.ivf_nprobe_fraction * nlist),
            )
            search_parameters["ivf.nprobe"] = nprobe
            if self.quantizer_ef_search_per_nprobe is not None:
                search_parameters["ivf.quantizer.hnsw.efSearch"] = (
                    self.quantizer_ef_search_per_nprobe * nprobe
                )
        if self.k_factor is not None:
            search_parameters["k_factor"] = self.k_factor

        return ResolvedIndexProfile(
            key=self.key,
            factory_description=factory_description,
            nlist=nlist,
            training_sample_size=training_sample_size,
            search_parameters=search_parameters,
        )


INDEX_PROFILES = {
    "flat": IndexProfile(key="flat", factory_template="Flat"),
    "sq8": IndexProfile(
        key="sq8", factory_template="SQ8", training_sample_cap=5_000
    ),
    "rabitq-refine-sq8": IndexProfile(
        key="rabitq-refine-sq8",
        factory_template="RR,RaBitQfs1,Refine(SQ8)",
        training_sample_cap=5_000,
    ),
    "hnsw32": IndexProfile(
        key="hnsw32",
        factory_template="HNSW32",
        training_sample_cap=5_000,
        hnsw_ef_search_per_k=3,
    ),
    "ivf-hnsw-rabitq-refine-sq8": IndexProfile(
        key="ivf-hnsw-rabitq-refine-sq8",
        factory_template="RR,IVF{nlist}_HNSW,RaBitQfs1,Refine(SQ8)",
        training_samples_per_list=40,
        nlist_scale=1.5,
        ivf_nprobe_fraction=0.05,
        ivf_nprobe_minimum=10,
        quantizer_ef_search_per_nprobe=2,
        k_factor=15,
    ),
}


def get_index_profile(profile: str | IndexProfile) -> IndexProfile:
    if isinstance(profile, IndexProfile):
        return profile
    try:
        return INDEX_PROFILES[profile]
    except KeyError as error:
        valid_profiles = ", ".join(INDEX_PROFILES)
        raise ValueError(
            f"Unknown index profile {profile!r}. Choose from: {valid_profiles}."
        ) from error