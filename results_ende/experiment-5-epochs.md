# Experiment: English -> German, 5 epochs

Checkpoint: `checkpoints/multi30k_en_de_5ep.pt`

| metric | value |
|--------|------:|
| validation loss (label-smoothed) | 1.8854 |
| validation perplexity | 8.80 |
| BLEU — validation (1014 sents) | 29.10 |
| BLEU — test (1000 sents) | 29.61 |
| wall-clock to reach this epoch | 29.9 min |

## Sample translations

| # | source (de) | reference (en) | model output (en) |
|--:|-------------|----------------|-------------------|
| 1 | A group of men are loading cotton onto a truck | Eine Gruppe von Männern lädt Baumwolle auf einen Lastwagen | Eine Gruppe Männer kauft Lebensmittel auf einem Lkw . |
| 2 | A man sleeping in a green room on a couch. | Ein Mann schläft in einem grünen Raum auf einem Sofa. | Ein Mann schläft in einem grünen Raum auf einem Sofa . |
| 3 | A boy wearing headphones sits on a woman's shoulders. | Ein Junge mit Kopfhörern sitzt auf den Schultern einer Frau. | Ein Junge mit Kopfhörern sitzt auf der Schultern einer Frau . |
| 4 | Two men setting up a blue ice fishing hut on an iced over lake | Zwei Männer bauen eine blaue Eisfischerhütte auf einem zugefrorenen See auf | Zwei Männer machen eine blaue Bluse auf einem See und einer blauen Hütte . |
| 5 | A balding man wearing a red life jacket is sitting in a small boat. | Ein Mann mit beginnender Glatze, der eine rote Rettungsweste trägt, sitzt in einem kleinen Boot. | Ein Mann mit nacktem Oberkörper , der eine rote Jacke trägt , sitzt in einem kleinen Boot . |
| 6 | A lady in a red coat, holding a bluish hand bag likely of asian descent, jumping off the ground for a snapshot. | Eine Frau in einem rotem Mantel, die eine vermutlich aus Asien stammende Handtasche in einem blauen Farbton hält, springt für einen Schnappschuss in die Luft. | Eine Dame in einem roten Mantel hält einen Teller mit der Hand und macht einen Teller auf dem Boden für einen Fußball . |
| 7 | A brown dog is running after the black dog. | Ein brauner Hund rennt dem schwarzen Hund hinterher. | Ein brauner Hund läuft nach dem schwarzen Hund nach . |
| 8 | A young boy wearing a Giants jersey swings a baseball bat at an incoming pitch. | Ein kleiner Junge mit einem Giants-Trikot schwingt einen Baseballschläger in Richtung eines ankommenden Balls. | Ein kleiner Junge in einem Trikot schlägt einen Baseball auf einem Ball in einem Wurf des Balls . |
