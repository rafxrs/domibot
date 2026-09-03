from ..card import Card
from ..enums import CardType

COPPER = Card("Copper", cost=0, types=(CardType.TREASURE,), coin_value=1)
SILVER = Card("Silver", cost=3, types=(CardType.TREASURE,), coin_value=2)
GOLD = Card("Gold", cost=6, types=(CardType.TREASURE,), coin_value=3)

ESTATE = Card("Estate", cost=2, types=(CardType.VICTORY,), vp_value=1)
DUCHY = Card("Duchy", cost=5, types=(CardType.VICTORY,), vp_value=3)
PROVINCE = Card("Province", cost=8, types=(CardType.VICTORY,), vp_value=6)

CURSE = Card("Curse", cost=0, types=(CardType.CURSE,), vp_value=-1)

BASIC_CARDS: dict[str, Card] = {
    c.name: c for c in [COPPER, SILVER, GOLD, ESTATE, DUCHY, PROVINCE, CURSE]
}
