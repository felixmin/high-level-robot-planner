1. What filters actually is

Right now filters is passed straight into SceneFilter(filters) and used like this:

for key, condition in self.filters.items():
    ...
    elif isinstance(condition, tuple) and len(condition) == 2:
        op, threshold = condition
        ...


So filters are scene-level, and each entry is:

Equality – "stabilized_label": "dynamic"

Comparison – ("max_trans": (">", 0.1)), etc.

Boolean – "contains_hand_sam3": True

Exclusion – ("label": ("!=", "static"))

Later, if you want pair-level filters, that’s an extra pass on self.pairs, but right now everything is scene-based.

2. How to write filters in your YAML

The main catch:
Python code checks isinstance(condition, tuple), but YAML gives you lists, not tuples, unless you use OmegaConf tricks.

So either:

(Recommended) in Python change:

elif isinstance(condition, tuple) and len(condition) == 2:


to:

elif isinstance(condition, (tuple, list)) and len(condition) == 2:


Then YAML lists work fine.

Once you do that, you can write filters like this:

Example 1: Only stabilized scenes
filters:
  stabilized_label: "stabilized"

Example 2: Only scenes with enough translation
filters:
  stabilized_label: "stabilized"
  stabilized_max_trans: [">", 0.05]   # condition: value > 0.05

Example 3: Exclude static scenes, only with a hand
filters:
  label: ["!=", "static"]
  contains_hand_sam3: true

Example 4: Require at least some rotation
filters:
  stabilized_label: "dynamic"
  stabilized_max_angle: [">=", 5.0]


Put that straight into your laq_pairs config:

# Metadata filtering (optional)
use_metadata: true
return_metadata: false
filters:
  stabilized_label: "stabilized"
  label: ["!=", "static"]
  stabilized_max_trans: [">", 0.05]
min_frames: 2


If you don’t change SceneFilter to accept lists, then ["!=", "static"] won’t be recognized as a tuple and will fall back to equality (which is wrong). So I’d really do that tiny code change.