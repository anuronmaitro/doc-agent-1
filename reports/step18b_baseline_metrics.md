# Step 18b — OCR repair re-run (BEFORE / AFTER numbers)

Reader: `facebook/nougat-base` (**pretrained, not fine-tuned** -- Step 18b fixes the inference path, not the model), revision `abfecedbb34367c820e233f710fdc7f54e6ab249`.

Run mode: `full` · repo commit `2b0f7209e017` · GPU `Tesla T4`.


## The four numbers plan.md Step 18b asks for

| metric | BEFORE (Step 16) | AFTER (this run) |
|---|---:|---:|
| no-output rate | 42.9% | 28.5% |
| median page coverage vs PDF text layer | 28.0% | 43.7% |
| book-wide word coverage vs PDF text layer | 15.1% | 28.3% |
| mean char-F1 (3 gold pages) | n/a (see per-page) | 0.3905 |
| weighted exact-formula-match (3 gold pages) | n/a (see per-page) | 0.0000 |
| as_p0360 precision / recall | 0.813 / 0.280 | 0.458 / 0.833 |

## Per-page baseline on the 3 A1 gold pages

| page | printed | char-F1 | precision | recall | exact-formula | gold formulas | pred formulas |
|---|---:|---:|---:|---:|---:|---:|---:|
| `as_p0243` | 243 | 0.3360 | 0.2734 | 0.4359 | 0.0000 | 0 | 0 |
| `as_p0255` | 255 | 0.2446 | 0.8763 | 0.1421 | 0.0000 | 14 | 3 |
| `as_p0360` | 360 | 0.5910 | 0.4579 | 0.8334 | 0.0000 | 13 | 18 |

**Aggregate:** mean char-F1 **0.3905** · mean precision **0.5359** · mean recall **0.4705** · exact-formula-match **0.0000** (weighted by formula count; unweighted 0.0000 across 27 formulas on 3 pages).


## Coverage and honest failures

- content pages in corpus: **1040**
- pages processed this run: **1040**
- transcripts produced: **744**
- degenerate/failed pages: **296**
- words from OUR OCR: **161941**
- regions detected: **6235** {'figure': 357, 'heading': 487, 'table': 1190, 'text': 4201}

Failed pages: `as_p0100`, `as_p1028`, `as_p0002`, `as_p0011`, `as_p0018`, `as_p0028`, `as_p0032`, `as_p0034`, `as_p0036`, `as_p0038`, `as_p0040`, `as_p0043`, `as_p0044`, `as_p0046`, `as_p0050`, `as_p0052`, `as_p0056`, `as_p0058`, `as_p0062`, `as_p0078`, `as_p0081`, `as_p0101`, `as_p0102`, `as_p0103`, `as_p0104`, `as_p0106`, `as_p0110`, `as_p0116`, `as_p0120`, `as_p0122`, `as_p0123`, `as_p0125`, `as_p0127`, `as_p0128`, `as_p0131`, `as_p0136`, `as_p0138`, `as_p0139`, `as_p0140`, `as_p0142`, `as_p0143`, `as_p0144`, `as_p0148`, `as_p0150`, `as_p0152`, `as_p0156`, `as_p0160`, `as_p0161`, `as_p0165`, `as_p0166`, `as_p0169`, `as_p0170`, `as_p0178`, `as_p0180`, `as_p0181`, `as_p0184`, `as_p0186`, `as_p0189`, `as_p0190`, `as_p0192`, `as_p0194`, `as_p0196`, `as_p0197`, `as_p0198`, `as_p0200`, `as_p0202`, `as_p0203`, `as_p0204`, `as_p0205`, `as_p0206`, `as_p0208`, `as_p0209`, `as_p0210`, `as_p0211`, `as_p0212`, `as_p0213`, `as_p0217`, `as_p0218`, `as_p0219`, `as_p0220`, `as_p0221`, `as_p0222`, `as_p0228`, `as_p0239`, `as_p0240`, `as_p0241`, `as_p0242`, `as_p0244`, `as_p0246`, `as_p0247`, `as_p0248`, `as_p0249`, `as_p0256`, `as_p0260`, `as_p0268`, `as_p0269`, `as_p0270`, `as_p0271`, `as_p0272`, `as_p0274`, `as_p0277`, `as_p0278`, `as_p0280`, `as_p0281`, `as_p0283`, `as_p0284`, `as_p0286`, `as_p0287`, `as_p0288`, `as_p0291`, `as_p0313`, `as_p0318`, `as_p0319`, `as_p0322`, `as_p0324`, `as_p0327`, `as_p0328`, `as_p0335`, `as_p0342`, `as_p0344`, `as_p0346`, `as_p0348`, `as_p0353`, `as_p0369`, `as_p0374`, `as_p0390`, `as_p0392`, `as_p0394`, `as_p0396`, `as_p0397`, `as_p0400`, `as_p0401`, `as_p0402`, `as_p0404`, `as_p0405`, `as_p0406`, `as_p0408`, `as_p0410`, `as_p0412`, `as_p0416`, `as_p0417`, `as_p0418`, `as_p0420`, `as_p0427`, `as_p0428`, `as_p0430`, `as_p0431`, `as_p0432`, `as_p0452`, `as_p0457`, `as_p0461`, `as_p0462`, `as_p0464`, `as_p0466`, `as_p0468`, `as_p0469`, `as_p0470`, `as_p0471`, `as_p0472`, `as_p0473`, `as_p0474`, `as_p0475`, `as_p0478`, `as_p0480`, `as_p0484`, `as_p0485`, `as_p0486`, `as_p0488`, `as_p0492`, `as_p0516`, `as_p0517`, `as_p0518`, `as_p0534`, `as_p0543`, `as_p0546`, `as_p0552`, `as_p0572`, `as_p0583`, `as_p0584`, `as_p0585`, `as_p0601`, `as_p0609`, `as_p0610`, `as_p0612`, `as_p0614`, `as_p0616`, `as_p0617`, `as_p0618`, `as_p0620`, `as_p0622`, `as_p0640`, `as_p0647`, `as_p0648`, `as_p0651`, `as_p0653`, `as_p0655`, `as_p0659`, `as_p0660`, `as_p0662`, `as_p0665`, `as_p0673`, `as_p0676`, `as_p0677`, `as_p0678`, `as_p0680`, `as_p0682`, `as_p0702`, `as_p0703`, `as_p0704`, `as_p0707`, `as_p0713`, `as_p0714`, `as_p0715`, `as_p0716`, `as_p0718`, `as_p0719`, `as_p0726`, `as_p0729`, `as_p0761`, `as_p0762`, `as_p0763`, `as_p0764`, `as_p0765`, `as_p0769`, `as_p0774`, `as_p0793`, `as_p0794`, `as_p0797`, `as_p0798`, `as_p0801`, `as_p0809`, `as_p0812`, `as_p0828`, `as_p0829`, `as_p0830`, `as_p0835`, `as_p0836`, `as_p0838`, `as_p0840`, `as_p0843`, `as_p0845`, `as_p0846`, `as_p0847`, `as_p0851`, `as_p0853`, `as_p0857`, `as_p0859`, `as_p0871`, `as_p0873`, `as_p0877`, `as_p0878`, `as_p0888`, `as_p0894`, `as_p0895`, `as_p0900`, `as_p0902`, `as_p0903`, `as_p0905`, `as_p0906`, `as_p0910`, `as_p0911`, `as_p0916`, `as_p0920`, `as_p0922`, `as_p0929`, `as_p0933`, `as_p0940`, `as_p0943`, `as_p0949`, `as_p0955`, `as_p0966`, `as_p0968`, `as_p0969`, `as_p0973`, `as_p0976`, `as_p0978`, `as_p0984`, `as_p0985`, `as_p0999`, `as_p1000`, `as_p1003`, `as_p1004`, `as_p1005`, `as_p1007`, `as_p1008`, `as_p1009`, `as_p1017`, `as_p1023`, `as_p1031`, `as_p1032`, `as_p1035`, `as_p1037`, `as_p1039`, `as_p1041`, `as_p1043`, `as_p1048`

## Timings

- corpus_render: 19.8 min
- loader: 0.9 min
- preprocess: 6.8 min
- layout: 4.2 min
- ocr_gated_subset: 8.2 min
- ocr_full_book: 665.9 min
- total: 706.7 min
