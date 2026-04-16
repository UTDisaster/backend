from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PromptVersion = Literal["v1", "v2", "v3", "v4"]


@dataclass(frozen=True)
class Prompt:
    version: PromptVersion
    system_instruction: str
    user_instruction: str


SYSTEM_INSTRUCTION = (
    "You are a trained satellite damage assessor for the xBD disaster assessment dataset. "
    "You compare pre- and post-disaster satellite imagery of a single building footprint "
    "and classify the damage using the xBD 4-level scale. "
    "You return strict JSON matching the provided schema, and nothing else."
)

DAMAGE_RUBRIC = """You are classifying damage to ONE specific building centered in the crop.
Only damage to THE STRUCTURE counts. Floodwater or debris near the building is
context, not damage. Minor alignment or lighting differences between pre and
post are NORMAL (different satellite passes) and are not damage.

xBD damage scale (choose exactly one):
- 0 no-damage: the building in the crop looks the same in pre and post.
  Unchanged roof, walls, outline. Water or debris AROUND it doesn't count.
- 1 minor-damage: subtle change to the building itself - small blue tarp, partial
  roof discoloration, localized debris ON the structure, shallow flood touching
  the building wall, visible but minor damage to <25% of the roof.
- 2 major-damage: visible structural compromise - hole in roof, partial wall
  collapse, large burn area on the building, building partially inundated with
  roof still visible, 25-75% of roof damaged or missing.
- 3 destroyed: the building is GONE or in ruins - bare foundation/slab, rubble
  pile with no standing walls, completely burned, fully submerged with no roof
  visible, or roof collapsed onto itself.

When uncertain between two adjacent levels, pick the LOWER one.
Only call it unreadable if the post really is opaque (full cloud). Otherwise
classify based on the best visible evidence."""


V1_USER = (
    "Compare the PRE (before) and POST (after) satellite crops of one building.\n\n"
    + DAMAGE_RUBRIC
    + "\n\nReturn JSON: {\"score\": 0|1|2|3, \"label\": \"no-damage|minor-damage|major-damage|destroyed\", "
    "\"confidence\": 0.0-1.0, \"description\": short reason citing specific visual evidence, <= 200 chars}"
)


V2_USER = (
    "Compare the PRE and POST satellite crops. The target building is in the center.\n\n"
    + DAMAGE_RUBRIC
    + "\n\nCalibration (use these to anchor):\n"
    "- Roof color + outline unchanged => 0 no-damage, even if water or debris is near.\n"
    "- Small tarp, localized burn mark, minor roof debris ON the structure => 1 minor-damage.\n"
    "- Clear hole in roof, visible wall collapse, roof partially missing or buckled => 2 major-damage.\n"
    "- Only foundation/slab remains, rubble pile, building entirely underwater with no roof visible,\n"
    "  or fully burned to ground => 3 destroyed.\n"
    "- Post shows flooding AROUND an intact-looking building => 0 no-damage (water adjacency is not damage).\n\n"
    "Return JSON only: {\"score\": 0|1|2|3, \"label\": one of [no-damage, minor-damage, major-damage, destroyed], "
    "\"confidence\": 0.0-1.0, \"description\": name the specific visual difference on the building itself, <= 200 chars}"
)


V3_USER = (
    "Compare the PRE and POST satellite crops. The target building is in the center.\n\n"
    + DAMAGE_RUBRIC
    + "\n\nThink step by step inside the description field. Format exactly:\n"
    "'pre: <building state, 2-5 words>. post: <building state, 2-5 words>. "
    "change: <what changed on the building, 2-8 words>. => score N'\n"
    "Only describe the BUILDING, not surrounding context. Then choose score/label/confidence.\n\n"
    "Return JSON only: {\"score\": 0|1|2|3, \"label\": one of [no-damage, minor-damage, major-damage, destroyed], "
    "\"confidence\": 0.0-1.0, \"description\": the pre/post/change sentence as above, <= 200 chars}"
)


V4_SYSTEM = (
    "You are a trained flood-damage assessor for Hurricane Florence. "
    "All damage is flood damage from storm surge or river overtopping - not wind or fire. "
    "You are shown a pre-disaster and a post-disaster satellite crop of the SAME location. "
    "The target building is outlined in bright RED in both crops. "
    "Classify damage to that red-outlined building only. "
    "Return strict JSON matching the schema and nothing else."
)


V4_USER = (
    "Compare the PRE and POST satellite crops. Classify the RED-OUTLINED building only.\n\n"
    "Hurricane Florence flood-damage rubric (apply strictly to the red-outlined building):\n\n"
    "- 0 no-damage: the red-outlined building looks the same pre and post. Dry ground, or\n"
    "  water far from the outline, or water touching only the outer wall at ground level.\n"
    "  Roof, walls, and outline unchanged. Flooding at other buildings does NOT count.\n\n"
    "- 1 minor-damage: the red-outlined building is intact but shows flood contact - shallow\n"
    "  water clearly entering around it, small debris deposited against it, or water up to\n"
    "  about knee-height of first floor. Roof unchanged, walls visibly above water.\n\n"
    "- 2 major-damage: the red-outlined building is partially inundated - water reaches\n"
    "  first-floor window level or covers most walls with roof still visible. OR the roof\n"
    "  has partial visible breach / collapse but the footprint is recognizable.\n\n"
    "- 3 destroyed: the red-outlined building is fully or mostly submerged (water at or\n"
    "  above roofline), roof has floated away or collapsed into the structure, building is\n"
    "  displaced from its foundation, OR only a bare slab remains where the building stood.\n\n"
    "Tie-breaking: when uncertain between adjacent levels, pick the LOWER one.\n"
    "Only minor alignment / lighting differences are normal and are not damage.\n\n"
    "Return JSON only: {\"score\": 0|1|2|3, \"label\": one of "
    "[no-damage, minor-damage, major-damage, destroyed], \"confidence\": 0.0-1.0, "
    "\"description\": one sentence naming the specific change to the RED-OUTLINED building, "
    "<= 200 chars}"
)


PROMPTS: dict[PromptVersion, Prompt] = {
    "v1": Prompt("v1", SYSTEM_INSTRUCTION, V1_USER),
    "v2": Prompt("v2", SYSTEM_INSTRUCTION, V2_USER),
    "v3": Prompt("v3", SYSTEM_INSTRUCTION, V3_USER),
    "v4": Prompt("v4", V4_SYSTEM, V4_USER),
}


def get_prompt(version: PromptVersion = "v2") -> Prompt:
    if version not in PROMPTS:
        raise ValueError(f"Unknown prompt version: {version}")
    return PROMPTS[version]
