from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling import ProposalNetwork


@META_ARCH_REGISTRY.register()
class OneStageDetector(ProposalNetwork):
    def forward(self, batched_inputs):
        if self.training:
            return super().forward(batched_inputs)
        processed_results = super().forward(batched_inputs)
        processed_results = [{"instances": r["proposals"]} for r in processed_results]
        return processed_results
