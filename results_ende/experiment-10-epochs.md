# Experiment: English -> German, 10 epochs

Checkpoint: `checkpoints/multi30k_en_de_10ep.pt`

| metric | value |
|--------|------:|
| validation loss (label-smoothed) | 1.7247 |
| validation perplexity | 6.86 |
| BLEU — validation (1014 sents) | 35.03 |
| BLEU — test (1000 sents) | 34.57 |
| wall-clock to reach this epoch | 59.9 min |

## Sample translations

| # | source (de) | reference (en) | model output (en) |
|--:|-------------|----------------|-------------------|
| 1 | A group of men are loading cotton onto a truck | Eine Gruppe von Männern lädt Baumwolle auf einen Lastwagen | Eine Gruppe Männer stellt sich auf einen Lkw auf einen Lkw auf . |
| 2 | A man sleeping in a green room on a couch. | Ein Mann schläft in einem grünen Raum auf einem Sofa. | Ein Mann schläft in einem grünen Raum auf einer Couch . |
| 3 | A boy wearing headphones sits on a woman's shoulders. | Ein Junge mit Kopfhörern sitzt auf den Schultern einer Frau. | Ein Junge mit Kopfhörern sitzt auf einer Frau . |
| 4 | Two men setting up a blue ice fishing hut on an iced over lake | Zwei Männer bauen eine blaue Eisfischerhütte auf einem zugefrorenen See auf | Zwei Männer stellen eine blaue Angel auf einem Holzboden eine aus . |
| 5 | A balding man wearing a red life jacket is sitting in a small boat. | Ein Mann mit beginnender Glatze, der eine rote Rettungsweste trägt, sitzt in einem kleinen Boot. | Ein Mann mit Glatzenansatz sitzt in einer roten Schwimmweste auf einem kleinen Boot . |
| 6 | A lady in a red coat, holding a bluish hand bag likely of asian descent, jumping off the ground for a snapshot. | Eine Frau in einem rotem Mantel, die eine vermutlich aus Asien stammende Handtasche in einem blauen Farbton hält, springt für einen Schnappschuss in die Luft. | Eine Dame in einem roten Mantel hält einen Behälter mit einem Ausdruck der asiatischen Kleidung von einem Boden springt für eine Kleinigkeit nach vorn . |
| 7 | A brown dog is running after the black dog. | Ein brauner Hund rennt dem schwarzen Hund hinterher. | Ein brauner Hund rennt nach dem schwarzen Hund hinterher . |
| 8 | A young boy wearing a Giants jersey swings a baseball bat at an incoming pitch. | Ein kleiner Junge mit einem Giants-Trikot schwingt einen Baseballschläger in Richtung eines ankommenden Balls. | Ein kleiner Junge , der ein Trikot trägt , schlägt einen Baseball auf einen Ball in einem Ball . |
