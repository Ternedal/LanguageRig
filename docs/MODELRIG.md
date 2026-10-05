# Kobling til ModelRig

LanguageRig og [ModelRig](https://github.com/Ternedal/ModelRig) har separate
installationer. ModelRig skal ikke importere LanguageRig som Python-afhængighed.
LanguageRig forbereder data og modeller; ModelRig bruger den lokale Ollama og
sit eksisterende RAG-index til chat og dokumentopslag.

| Forløb | LanguageRig | Eksisterende grænseflade |
| --- | --- | --- |
| Trænet model | Merge, operatørkonverteret GGUF og lokal registrering | Ollama create og /api/tags; ModelRigs modelvælger |
| Bogopslag | Kapitel-/sidebaseret eksport og publish-rag | POST /api/v1/rag/ingest med documents, chunk_size og chunk_overlap |
| Direkte modelsammenligning | Baseline/kandidat, modeldigests og HTML | Lokal Ollama /api/tags og /api/chat |

## Samme Ollama-instans

Brug samme adresse som ModelRigs MODELRIG_OLLAMA_URL. CLI'en læser denne
miljøvariabel som standard for registrering og sammenligning; ellers bruges
http://127.0.0.1:11434. På en rig, der bruger port 11435, sættes adressen sådan:

~~~powershell
$env:MODELRIG_OLLAMA_URL = "http://127.0.0.1:11435"
~~~

~~~bash
export MODELRIG_OLLAMA_URL=http://127.0.0.1:11435
~~~

Ved brug fra WSL skal den valgte lokale Ollama-adresse kunne nås fra WSL.
En vellykket registrering gør kandidatnavnet tilgængeligt via Ollamas modeloversigt.
ModelRig skifter ikke standardmodel automatisk. Se [README](../README.md) for
merge, GGUF-konvertering, pakning og registrering.

## Bogopslag

Hvert document indeholder text og et stabilt source-ID. Alle tekststykker fra
samme kapitel/side sendes samlet, så genudsendelse kan erstatte den eksisterende
kilde. Publish bruger et eksisterende parret enhedstoken fra
MODELRIG_DEVICE_TOKEN; LanguageRig foretager ingen ny parring og gemmer ikke
tokenet i eksport eller kvittering.

ModelRig-serverens baseadresse gives eksplicit til publish-rag, hvis den
afviger fra http://127.0.0.1:8080. Eksporten og API-kontrakten er testet mod
en lokal HTTP-fixture. Den rigtige rig skal stadig afprøves.

## Status og evaluering

Trained betyder, at træningskørslen er afsluttet. Registered betyder, at
Ollama har registreret kandidaten. En afsluttet comparison betyder, at
de forventede modelsvar er modtaget og gemt. Disse tilstande dokumenterer
ikke i sig selv bedre dansk, faglig korrekthed eller fungerende værktøjskald.

GPU-træning, GGUF-konvertering, modelkvalitet og klientadfærd afprøves på
riggen. Den direkte sammenligning bruger ikke ModelRigs RAG; en separat
RAG-måling skal give baseline og kandidat samme dokumentadgang.

## Flytning og kilde

Piloten er flyttet fra ModelRig-kladden
[PR #2091](https://github.com/Ternedal/ModelRig/pull/2091),
kildecommit ef38c200b7372b4046627d8a313bcb2d699f07da.
Python-koden, tests og modelkonfigurationen er bevaret fra denne commit;
installationsstier, CI og dokumentationslinks er tilpasset det selvstændige repo.
Den oprindelige MIT-licens er bevaret.
