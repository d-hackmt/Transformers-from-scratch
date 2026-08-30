# Experiment: English -> German, 15 epochs

Checkpoint: `checkpoints/multi30k_en_de_15ep.pt`

| metric | value |
|--------|------:|
| validation loss (label-smoothed) | 1.7942 |
| validation perplexity | 7.48 |
| BLEU — validation (1014 sents) | 34.70 |
| BLEU — test (1000 sents) | 32.74 |
| wall-clock to reach this epoch | 89.8 min |

## Sample translations

| # | source (de) | reference (en) | model output (en) |
|--:|-------------|----------------|-------------------|
| 1 | A group of men are loading cotton onto a truck | Eine Gruppe von Männern lädt Baumwolle auf einen Lastwagen | Eine Gruppe Männer stellt sich gegenseitig auf einen Lkw vor . |
| 2 | A man sleeping in a green room on a couch. | Ein Mann schläft in einem grünen Raum auf einem Sofa. | Ein Mann schläft in einem grünen Zimmer auf einem Sofa . |
| 3 | A boy wearing headphones sits on a woman's shoulders. | Ein Junge mit Kopfhörern sitzt auf den Schultern einer Frau. | Ein Junge mit Kopfhörern sitzt auf den Schultern einer Frau . |
| 4 | Two men setting up a blue ice fishing hut on an iced over lake | Zwei Männer bauen eine blaue Eisfischerhütte auf einem zugefrorenen See auf | Zwei Männer stellen eine blaue und decken den See mit ihren Fang . |
| 5 | A balding man wearing a red life jacket is sitting in a small boat. | Ein Mann mit beginnender Glatze, der eine rote Rettungsweste trägt, sitzt in einem kleinen Boot. | Ein kahl werdender Mann mit einer roten Schwimmweste sitzt in einem kleinen Boot . |
| 6 | A lady in a red coat, holding a bluish hand bag likely of asian descent, jumping off the ground for a snapshot. | Eine Frau in einem rotem Mantel, die eine vermutlich aus Asien stammende Handtasche in einem blauen Farbton hält, springt für einen Schnappschuss in die Luft. | Eine Frau in einem roten Mantel , die einen Umschlag in der Hand hält , Asiaten um den Boden herum afrikanischer Herkunft , springt auf dem Boden . |
| 7 | A brown dog is running after the black dog. | Ein brauner Hund rennt dem schwarzen Hund hinterher. | Ein brauner Hund rennt dem schwarzen Hund nach . |
| 8 | A young boy wearing a Giants jersey swings a baseball bat at an incoming pitch. | Ein kleiner Junge mit einem Giants-Trikot schwingt einen Baseballschläger in Richtung eines ankommenden Balls. | Ein Junge , der die das Gleichgewicht trägt , schlägt einen Baseballschläger vom nächsten Schlag ab . |
