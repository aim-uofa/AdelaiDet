adet.modeling package
===========================

.. automodule:: adet.modeling
    :members:
    :undoc-members:
    :show-inheritance:

adet.modeling.backbone module
---------------------------------------

.. automodule:: adet.modeling.backbone
    :members:
    :undoc-members:
    :show-inheritance:

adet.modeling.poolers module
---------------------------------------

.. automodule:: adet.modeling.poolers
    :members:
    :undoc-members:
    :show-inheritance:


Model Registries
-----------------

These are different registries provided in modeling.
Each registry provide you the ability to replace it with your customized component,
without having to modify detectron2's code.

Note that it is impossible to allow users to customize any line of code directly.
Even just to add one line at some place,
you'll likely need to find out the smallest registry which contains that line,
and register your component to that registry.


* detectron2.modeling.META_ARCH_REGISTRY
* detectron2.modeling.BACKBONE_REGISTRY
* detectron2.modeling.PROPOSAL_GENERATOR_REGISTRY
* detectron2.modeling.RPN_HEAD_REGISTRY
* detectron2.modeling.ANCHOR_GENERATOR_REGISTRY
* detectron2.modeling.ROI_HEADS_REGISTRY
* detectron2.modeling.ROI_BOX_HEAD_REGISTRY
* detectron2.modeling.ROI_MASK_HEAD_REGISTRY
* detectron2.modeling.ROI_KEYPOINT_HEAD_REGISTRY