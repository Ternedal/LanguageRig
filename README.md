# LanguageRig 0.1

Et selvstændigt, lokalt Python-modul til at videreuddanne en eksisterende dansk
sprogmodel på EPUB/PDF-materiale og bruge resultatet i ModelRig/Kaliv.
LanguageRig ligger i sit eget repo og installeres separat fra ModelRig.

Første version har en CLI. Den indeholder bogimport, metadata/udvælgelse,
datasæt med opdeling efter værk, en valgfri QLoRA-runner, checkpoint-resume,
adapter-merge, GGUF-pakning/Ollama-registrering og sammenligning af modelversioner.
Bogteksten kan også eksporteres til ModelRigs eksisterende RAG-API.

Til første kørsel på riggen findes et samlet
[Windows/WSL-pilotforløb](docs/PILOT.md): bogimport, datasæt og konfiguration
med prepare-pilot, lokalt miljøtjek med doctor og eksplicit træningsstart.

Der er endnu ingen grafisk bogvælger, automatisk generering af træningsdialoger
eller OCR. GPU-træning og kvalitet på virkelige danske bøger skal kvalificeres
på riggen; syntetiske import- og transporttests dokumenterer ikke modelkvalitet.

## Installation og import på Windows

Klon LanguageRig, og kør fra repoets rod:

~~~powershell
git clone https://github.com/Ternedal/LanguageRig.git
Set-Location .\LanguageRig
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -e .
& .\.venv\Scripts\languagerig.exe --workspace .\data init
& .\.venv\Scripts\languagerig.exe --workspace .\data import "D:\Boeger\Dansk" --training-allowed
& .\.venv\Scripts\languagerig.exe --workspace .\data inventory
~~~

EPUB læses i spine-rækkefølge. Titel, forfatter, sprog, kapitler og kilde-ID
bevares. PDF læses som tekst pr. side, og gentagne sidehoveder/-fødder fjernes.
Scannede PDF'er kræver en separat OCR-behandling; tom tekst tæller ikke som import.
Fejl rapporteres pr. bog, så én dårlig fil ikke stopper hele mappen.
Genimport af samme fil er idempotent. Originalerne ændres ikke.

Import er en lokal læsning, også uden markering til træning.
Flaget --training-allowed registrerer operatørens valg af materiale, der kan
bruges til træning; det er ikke en automatisk vurdering af brugsrettigheder.
EPUB-sprog læses fra metadata. PDF-sprog er ukendt indtil det mærkes.
Brug --language da ved import af en mappe, hvis dens danske sprog er kontrolleret.

Kopiér det fulde source_id fra inventory for at mærke enkelte bøger:

~~~powershell
& .\.venv\Scripts\languagerig.exe label SOURCE_ID --language da --genre nonfiction --topic arkitektur
& .\.venv\Scripts\languagerig.exe label SOURCE_ID --genre fiction
& .\.venv\Scripts\languagerig.exe label SOURCE_ID --work-id faelles-vaerk-id
~~~

Flere udgaver/oversættelser af ét værk bør få samme work-id. Automatisk gruppering
bruger titel+forfatter og identisk normaliseret tekst. Næsten ens tekster med
ændrede metadata kræver stadig manuel gruppering. Genre/emne sættes manuelt i v0.1.

## Datasæt og første træningsplan

~~~powershell
& .\.venv\Scripts\languagerig.exe dataset dansk-pilot
& .\.venv\Scripts\languagerig.exe verify .\data\datasets\dansk-pilot
& .\.venv\Scripts\languagerig.exe train .\examples\munin-pilot.json
~~~

Mindst tre forskellige, udvalgte danske værker med accepteret tekstkvalitet
kræves til train/validation/test. Samme værk og identiske bogtekster forbindes
før opdeling. Identiske passager, der forekommer på tværs af splits, fjernes.
Datasæt er navngivne snapshots med kildeopdeling og checksums; en ny version
skal have et nyt navn. Ændrede data afvises før træning.

Uden --execute viser train kun træningsplanen; kommandoen downloader ingen
modeller og starter ingen GPU-job.
Eksemplet bruger Munin Apertus 8B som kandidat, ikke som en dokumenteret
vinder eller en kvalificeret 12 GB-træningsprofil. Dens ApertusForCausalLM-
arkitektur passer til runnerens AutoModelForCausalLM. Munin Ministral 8B bruger
Mistral3ForConditionalGeneration og kræver en særskilt modelintegration.

Konfigurationsstier er relative til JSON-filens mappe. Ved brug af en anden
datamappe skal dataset/output_dir derfor ændres i en kopi af konfigurationen.
max_steps er pilotens stopgrænse og har forrang over epochs.

En fagprofil bygges med --genre nonfiction og/eller --topic EMNE.
Et bredt danskkorpus kan inkludere både skønlitteratur og fagbøger.
En tekstprofil træner næste-token-forudsigelse; den er ikke i sig selv
assistenttræning.

### Kvalitetstjekkede dialoger

Dataset --instructions sti.jsonl bygger i stedet et instruktionsdatasæt.
Hver linje skal have reviewed=true, source_ids med importerede bøgers fulde
ID'er og messages, der følger system? / user / assistant / ... / assistant.
Eksempler fra flere bøger må kun kombinere kilder i samme split.
Samme seed/kildeudvalg som tekstsnapshot giver samme kildeopdeling.

~~~json
{
  "reviewed": true,
  "source_ids": ["ERSTAT_MED_ET_FULDT_KILDE_ID"],
  "messages": [
    {"role": "user", "content": "Forklar bogens definition af begrebet."},
    {"role": "assistant", "content": "Et kontrolleret dansk svar baseret på kapitlet."}
  ]
}
~~~

Der fremstilles ingen dialoger automatisk. Instruction-runneren beregner loss
på den sidste assistentbesvarelse og maskerer prompten. Eksempler, der er for
lange, afvises frem for at afkorte svaret tavst.

## QLoRA i separat WSL-miljø

Brug en separat Linux-venv ved siden af Windows-venv'en. Installér en CUDA-version
af PyTorch, der passer til riggens driver, før træning. Fra LanguageRig-repoets rod
i WSL:

~~~bash
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
python -m pip install -e '.[train]'
languagerig train examples/munin-pilot.json
~~~

Før træning skal readiness-forløbet køres fra PowerShell i repoets rod på
Windows. Det kører doctor og fit-probe i WSL med .venv-wsl og skriver
data\checks\train-doctor.json, fit-probe.json og readiness.json, som --execute
kræver. Readiness-kvitteringen dannes kun af scripts/check-pilot-ready.ps1; der
findes ingen CLI-kommando til den:

~~~powershell
.\scripts\check-pilot-ready.ps1 -Config '.\examples\munin-pilot.json' -Gpu 0
~~~

(Pilotforløbet i [docs/PILOT.md](docs/PILOT.md) bruger i stedet
data\pilots\NAME\configs\NAME.json.) Scriptet bruger WSL-distributionen
Ubuntu-22.04, medmindre -Distribution angives, og .venv-wsl/bin/python i
repoet, medmindre -WslPython angives.

Scriptet kører doctor og fit-probe i rækkefølge og udskriver et samlet READY-resumé
med GPU, compute capability, samlet VRAM, minimum observeret fri VRAM,
modelrevision og sekvenslængde. Derefter startes træningen i WSL:

~~~bash
export CUDA_VISIBLE_DEVICES=0
languagerig train examples/munin-pilot.json --execute \
  --fit-report data/checks/fit-probe.json \
  --readiness-report data/checks/readiness.json \
  --doctor-report data/checks/train-doctor.json
~~~

eller fra PowerShell med samme config og GPU:

~~~powershell
.\scripts\train-pilot.ps1 -Config '.\examples\munin-pilot.json' -Gpu 0 -Execute
~~~

Kør ikke doctor eller fit-probe igen mellem readiness og træningsstart; de
overskriver rapporterne, som kvitteringen hasher.

Runneren bruger én synlig GPU, 4-bit NF4, LoRA på lineære lag, batch=1,
gradient accumulation og gradient checkpointing. En eksekverende træningskørsel
kræver en READY-kvittering samt den doctor- og fit-probe-rapport, som
kvitteringen hasher. Config- og datasæthash skal stadig matche; ændres nogen af
gate-rapporterne efter readiness, afvises træningsstart. Fit-proben låser desuden den konkrete Hugging Face-
commit, og træningen bruger præcis denne resolved revision, selv hvis et tag som
main flytter sig mellem probe og træningsstart. GPU-model, compute capability, samlet VRAM og de centrale runtime-versioner
(torch, transformers, peft, accelerate, bitsandbytes og datasets) bindes også
til gaten; ændres miljøet, kræves en ny fit-probe. To 12 GB-kort samles ikke
automatisk til én 24 GB-pulje. Stop øvrige modeller på det valgte kort og mål
VRAM/ydelse med et kort pilotjob. En mindre kompatibel dansk/multilingual model
kan vælges i konfigurationen, hvis kandidaten ikke passer.

Ved udførelse låses grundmodellens faktiske Hugging Face-revision i run.json.
Remote model code og ekstern træningstelemetri er slået fra.
progress.jsonl indeholder loss/step, og checkpoints gemmes løbende.
Testsplit bruges aldrig til træning eller checkpoint-valg.

~~~bash
languagerig train examples/munin-pilot.json --execute \
  --fit-report data/checks/fit-probe.json \
  --readiness-report data/checks/readiness.json \
  --doctor-report data/checks/train-doctor.json \
  --resume data/runs/dansk-pilot/checkpoint-20
~~~

Resume kræver samme konfiguration og uændret datasæt. Checkpointet skal ligge
i den oprindelige run-mappe. En frisk kørsel kræver en tom run-mappe.
Trænet status er adskilt fra dokumenteret kvalitetsforbedring.

## Fra adapter til ModelRig-model

~~~bash
languagerig merge data/runs/dansk-pilot data/merged/pilot
languagerig merge data/runs/dansk-pilot data/merged/pilot --execute
~~~

Merge bruger den præcise grundmodel/revision og køres i FP16 på CPU.
Konverter derefter med en kompatibel llama.cpp-checkout; dens egne Python-
afhængigheder skal være installeret. Tilpas værktøjsstierne:

~~~bash
python /path/to/llama.cpp/convert_hf_to_gguf.py data/merged/pilot \
  --outfile data/pilot-f16.gguf --outtype f16
/path/to/llama.cpp/build/bin/llama-quantize \
  data/pilot-f16.gguf data/pilot-q4.gguf Q4_K_M
languagerig package-model data/runs/dansk-pilot data/pilot-q4.gguf --name kaliv-dansk:pilot
languagerig register-model data/exports/kaliv-dansk-pilot --url http://127.0.0.1:11435
~~~

GGUF-konverteringen er et separat operatørtrin i v0.1. Pakken kontrollerer GGUF-
header og checksums, men beviser ikke, at en operatørleveret GGUF er konverteret
fra den angivne adapter. Modelfile og vægte verificeres før registrering.
Registrering gør kandidaten synlig i ModelRigs eksisterende modelvælger; den
skifter ikke standardmodellen og giver ingen produktionsgodkendelse.
Brug samme Ollama-port som MODELRIG_OLLAMA_URL på riggen.

## Sammenligning

Importér også den uændrede danske grundmodel i samme lokale Ollama, og brug
dens præcise modelnavn som baseline. Begge navne skal eksistere og have
forskellige modeldigests:

~~~bash
languagerig evaluate examples/basic-cases.jsonl data/evaluation/pilot \
  --baseline munin-apertus:baseline --candidate kaliv-dansk:pilot \
  --url http://127.0.0.1:11435 --dataset data/datasets/dansk-pilot
~~~

comparison.json gemmer modeldigests, svar, simple formatkontroller og varighed.
comparison.html viser svar side om side til menneskelig vurdering.
Bogafledte spørgsmål skal angive source_ids fra testsplit.
Eksempelsættet er kun generelle danske screeningsopgaver; supplér det med
kontrollerede fagspørgsmål fra de reserverede bøger.
Dette evaluerer direkte chat, ikke RAG eller rigtige værktøjskald.
Ændrede modeldigests, ufuldstændige svar og transportfejl giver failed.
Automatiske formatkontroller alene markerer aldrig modellen som forbedret.

## Bogbibliotek i ModelRigs RAG

~~~powershell
& .\.venv\Scripts\languagerig.exe export-rag .\data\exports\boeger-v1
# MODELRIG_DEVICE_TOKEN skal indeholde et eksisterende parret enhedstoken.
& .\.venv\Scripts\languagerig.exe publish-rag .\data\exports\boeger-v1 --url http://127.0.0.1:8080
~~~

Eksporten bruger samme documents/text/source-kontrakt som ModelRigs tools/modelrig-cli.py.
Hvert kapitel/side får et stabilt kilde-ID. Genudsendelse erstatter samme kilde
i eksisterende RAG, og hver kilde sendes i én atomisk API-anmodning.
En hel bogsamling er ikke én samlet transaktion; partial kvitteringer viser,
hvilke kilder der faktisk blev sendt. En genkørsel er idempotent pr. kilde.
Token gemmes ikke i kvitteringer eller eksport.

RAG-eksport er uafhængig af træningsvalg og kan omfatte testbøger. Evaluering
med dokumentopslag skal derfor planlægges som en særskilt måling med samme
dokumentadgang for baseline og kandidat.

## Test og lokal udvikling

Testene kræver kun basispakkerne; GPU-pakkerne i extra'en `train` er ikke
nødvendige. Fra repoets rod:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
languagerig --help
~~~

På Windows tester ./tests/test_scripts.ps1 (PowerShell 7.3+) pilotscripts og
WSL-argumenttransport med en stub. CI kører unittest og `languagerig --help` på
Windows og Linux med Python 3.10 og 3.12, og scripttesten på Windows.
Træningsafhængighederne installeres med `python -m pip install -e '.[train]'`
(se [QLoRA i separat WSL-miljø](#qlora-i-separat-wsl-miljø)).

## Dokumentation

| Dokument | Indhold |
| --- | --- |
| [docs/PILOT.md](docs/PILOT.md) | Første pilot på Windows og WSL: prepare-pilot, readiness og træningsstart |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Arkitektur og afgrænsning |
| [docs/MODELRIG.md](docs/MODELRIG.md) | Koblingen til ModelRig: Ollama-registrering, RAG-eksport og status |

## Plads i Kaliv

LanguageRig er et selvstændigt værktøj i Kaliv-økosystemet. Det har to
koblinger til [ModelRig](https://github.com/Ternedal/ModelRig), begge beskrevet
i [docs/MODELRIG.md](docs/MODELRIG.md): en trænet model registreres i den lokale
Ollama-instans, som ModelRig også bruger, og bøger kan sendes til ModelRigs RAG
med publish-rag (POST /api/v1/rag/ingest med et parret enhedstoken). ModelRig
kalder ikke LanguageRig.

Systemets ansvarsfordeling er defineret i
[ModelRig: docs/KALIV_SYSTEM_DEFINITION.md](https://github.com/Ternedal/ModelRig/blob/main/docs/KALIV_SYSTEM_DEFINITION.md).
De øvrige repositories er listet i
[Kaliv-repoets README](https://github.com/Ternedal/Kaliv#relaterede-repositories).
