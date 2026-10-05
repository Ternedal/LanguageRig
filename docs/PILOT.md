# Første pilot på Windows og WSL

Forløbet består af forberedelse på Windows, miljøtjek i WSL og et eksplicit
træningsjob. Forberedelse og miljøtjek downloader ingen modelvægte.
Træningskommandoen med -Execute kan hente den valgte grundmodel.

## Forbered bøgerne på Windows

Brug PowerShell 7.3 eller nyere, og kør fra LanguageRig-repoets rod.
Tilpas bogstien. Flaget -TrainingAllowed registrerer dit materialevalg.
Tilføj kun -Language da, hvis hele den valgte mappe er kontrolleret som dansk;
ellers bruges EPUB-metadata, og PDF-sproget skal mærkes manuelt.

~~~powershell
.\scripts\prepare-pilot.ps1 -Books 'D:\Boeger\Dansk' -Name dansk-pilot -TrainingAllowed -Language da
~~~

Scriptet opretter et separat Windows-miljø i .venv og installerer kun
LanguageRigs basispakker. Mindst tre forskellige danske værker kræves.
Hver pilot får sit eget workspace under data/pilots/NAME. Her gemmes importerede
tekster, et værkbaseret datasæt, konfiguration og rapporter:

| Fil | Indhold |
| --- | --- |
| configs/NAME.json | Konfiguration med portable relative datasæt-/run-stier |
| pilots/NAME.json | Importstatus, fejl, ekskluderede kilder og datasæthash |
| checks/prepare-doctor.json | Windows-miljø og kontrol af konfiguration/datasæt |
| datasets/NAME/manifest.json | Kildeopdeling og checksums |

Et genbrugt pilotnavn afvises. Fejl under import giver en rapport og en
fejlkode; hvis nok gode bøger findes, kan et gyldigt datasæt være forberedt
med status prepared_with_import_errors. Træning startes ikke af dette script.
Ret materialevalget, og brug et nyt pilotnavn, eller gennemgå det forberedte
datasæt og brug dets eksisterende konfiguration.

En eksisterende Python-installation kan vælges med -Python, en anden
projektmappe med -WorkspaceRoot og en kortere pilot med -MaxSteps.

## Installér WSL-miljøet én gang

Åbn Ubuntu-22.04, og gå til samme checkout gennem WSL-stien, eksempelvis:

LanguageRig kræver Python 3.10 eller nyere. Kontrollér python3 --version;
brug en passende Python-installation til venv'en.

~~~bash
cd /mnt/c/Users/admin/Desktop/LanguageRig
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
~~~

Installér en CUDA-udgave af PyTorch, der passer til driveren, efter
[PyTorchs installationsvejledning](https://pytorch.org/get-started/locally/).
Installér derefter LanguageRigs træningspakker:

~~~bash
python -m pip install -e '.[train]'
~~~

Et allerede fungerende WSL-miljø kan bruges med -WslPython /absolut/sti/bin/python.
Windows-venv'en kan ikke bruges som Linux-venv.

## Miljøtjek og træning

Kør fra PowerShell. Standardkørslen laver kun miljøtjek; -Gpu er kortets indeks
i den valgte WSL-installation. To kort skjules ikke som én samlet VRAM-pulje.

~~~powershell
.\scripts\train-pilot.ps1 -Config '.\data\pilots\dansk-pilot\configs\dansk-pilot.json' -Gpu 0
~~~

Miljøtjekket kontrollerer installerede pakkeversioner, det reserverede datasæt,
én synlig CUDA-GPU og en lille NF4-beregning i bitsandbytes. Rapporten viser
ledig/samlet VRAM og diskplads. training_environment_ready betyder, at disse
miljøkontroller er bestået; det beviser ikke, at 8B-kandidaten passer i VRAM,
eller at træningen forbedrer dansk. Det faktiske modeljob skal afprøves.

Ved et bestået miljøtjek bør modelens faktiske VRAM-fit måles før et længere job:

~~~powershell
.\scripts\train-pilot.ps1 -Config '.\data\pilots\dansk-pilot\configs\dansk-pilot.json' -Gpu 0 -FitProbe
~~~

-FitProbe må hente og loade den valgte grundmodel. Den kører præcis én
batch=1 forward/backward/AdamW-mikrostep ved konfigurationens fulde
max_seq_length, gemmer peak allocated/reserved VRAM i checks/fit-probe.json
og gemmer ingen adapter. En bestået probe viser kun, at denne ene mikrostep
passer; den beviser ikke langtidstabilitet eller bedre dansk modelkvalitet.

Når både miljøtjek og fit-probe er bestået og rapporten har
training_gate=pass, kan træning startes eksplicit. Launcher og CLI kontrollerer,
at rapportens config- og datasæthash matcher den aktuelle træningsplan.
Den konkrete Hugging Face-commit, som fit-proben loadede, gemmes som
resolved_revision og genbruges direkte af træningen, så et flyttet main-tag
ikke kan ændre modellen mellem probe og job. Den valgte GPU-model, compute capability, samlede VRAM og de centrale
træningspakkeversioner skal også være identiske med fit-proben; ellers blokeres
træningsstart og der skal køres en ny probe:

~~~powershell
.\scripts\train-pilot.ps1 -Config '.\data\pilots\dansk-pilot\configs\dansk-pilot.json' -Gpu 0 -Execute
~~~

Startscriptet gentager miljøtjekket, og en fejl stopper før vægtindlæsning.
Det vælger én GPU for hele WSL-processen og gemmer checks/train-doctor.json.
Brug -Gpu 1 til det andet kort og -Distribution, hvis distributionen hedder
noget andet. Stop andre modeljob på det valgte kort før et pilotjob.

~~~powershell
.\scripts\train-pilot.ps1 -Config '.\data\pilots\dansk-pilot\configs\dansk-pilot.json' -Gpu 0 -Execute -Resume '.\data\pilots\dansk-pilot\runs\dansk-pilot\checkpoint-20'
~~~

Resume kræver stadig samme konfiguration og uændret datasæt.
Træningsstatus og progress.jsonl ligger under runs/NAME.
model_load_status registrerer vægtindlæsning; faktisk netværksdownload kontra
cachegenbrug spores ikke i træningskørslen.

## Direkte CLI og afprøvning

~~~bash
languagerig doctor --gpu 0 --require-training --report data/checks/doctor.json
CUDA_VISIBLE_DEVICES=0 languagerig fit-probe data/configs/pilot-v2.json --execute --report data/checks/fit-probe.json
CUDA_VISIBLE_DEVICES=0 languagerig train data/configs/pilot-v2.json --execute --fit-report data/checks/fit-probe.json
languagerig --workspace data prepare-pilot /path/to/books --name pilot-v2 --training-allowed
~~~

Direkte prepare-pilot bruger alle træningsegnede bøger i det valgte workspace.
doctor --gpu påvirker kun miljøtjekkets underproces. Ved direkte train skal
CUDA_VISIBLE_DEVICES sættes for træningsprocessen som vist i README.

CI tester Python-forløbet på Windows/Linux, det rigtige forberedelsesscript på
Windows og WSL-argumenttransport med en kontrolleret stub. Bash-startens
stop/fit/execute/resume-forløb testes på Linux. Der køres ingen rigtige CUDA-job
eller WSL-distributioner i CI. Afprøvning på riggen mangler stadig.
