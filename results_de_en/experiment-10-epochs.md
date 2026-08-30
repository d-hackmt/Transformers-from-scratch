# Experiment: German -> English, 10 epochs

Checkpoint: `checkpoints/multi30k_de_en_10ep.pt`

| metric | value |
|--------|------:|
| validation loss (label-smoothed) | 1.6609 |
| validation perplexity | 6.42 |
| BLEU — validation (1014 sents) | 35.90 |
| BLEU — test (1000 sents) | 37.04 |
| wall-clock to reach this epoch | 56.1 min |

## Sample translations

| # | source (de) | reference (en) | model output (en) |
|--:|-------------|----------------|-------------------|
| 1 | Eine Gruppe von Männern lädt Baumwolle auf einen Lastwagen | A group of men are loading cotton onto a truck | A group of men loading into a truck . |
| 2 | Ein Mann schläft in einem grünen Raum auf einem Sofa. | A man sleeping in a green room on a couch. | A man sleeping on a couch in a green room . |
| 3 | Ein Junge mit Kopfhörern sitzt auf den Schultern einer Frau. | A boy wearing headphones sits on a woman's shoulders. | A young boy with headphones on his shoulder while sitting on a woman . |
| 4 | Zwei Männer bauen eine blaue Eisfischerhütte auf einem zugefrorenen See auf | Two men setting up a blue ice fishing hut on an iced over lake | Two men are building a blue Coca - Cola set at a lake . |
| 5 | Ein Mann mit beginnender Glatze, der eine rote Rettungsweste trägt, sitzt in einem kleinen Boot. | A balding man wearing a red life jacket is sitting in a small boat. | A balding man wearing a red life jacket sitting in a small boat . |
| 6 | Eine Frau in einem rotem Mantel, die eine vermutlich aus Asien stammende Handtasche in einem blauen Farbton hält, springt für einen Schnappschuss in die Luft. | A lady in a red coat, holding a bluish hand bag likely of asian descent, jumping off the ground for a snapshot. | A woman in a red coat , what looks like an old - fashioned handbag in a blue lighted mess on the air for a shot . |
| 7 | Ein brauner Hund rennt dem schwarzen Hund hinterher. | A brown dog is running after the black dog. | A brown dog runs towards the black dog . |
| 8 | Ein kleiner Junge mit einem Giants-Trikot schwingt einen Baseballschläger in Richtung eines ankommenden Balls. | A young boy wearing a Giants jersey swings a baseball bat at an incoming pitch. | A little boy with a batman belt is swinging a baseball ball toward an unseen ball . |
