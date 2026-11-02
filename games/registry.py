"""Game registry: the journey order (3 easy -> 1 medium -> 1 hard)."""
from games.game1_bubble_garden import BubbleGarden
from games.game2_lantern_lighter import LanternLighter
from games.game3_smile_sunshine import SmileSunshine
from games.game4_mirror_dancer import MirrorDancer
from games.game5_spell_caster import SpellCaster

GAMES = {1: BubbleGarden, 2: LanternLighter, 3: SmileSunshine, 4: MirrorDancer, 5: SpellCaster}
