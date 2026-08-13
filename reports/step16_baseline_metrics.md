# Step 16 — baseline OCR (BEFORE numbers)

Reader: `facebook/nougat-base` (**pretrained, not fine-tuned**), revision `abfecedbb34367c820e233f710fdc7f54e6ab249`.

Run mode: `full` · repo commit `1cd83a338c67` · GPU `Tesla T4`.


## Per-page baseline on the 3 A1 gold pages

| page | printed | char-F1 | exact-formula | gold formulas | pred formulas |
|---|---:|---:|---:|---:|---:|
| `as_p0360` | 360 | 0.4166 | 0.0000 | 13 | 5 |

**Aggregate:** mean char-F1 **0.4166** · exact-formula-match **0.0000** (weighted by formula count; unweighted 0.0000 across 13 formulas on 1 pages).


## Coverage and honest failures

- content pages in corpus: **1040**
- pages processed this run: **1040**
- transcripts produced: **594**
- degenerate/failed pages: **446**
- words from OUR OCR: **87523**
- regions detected: **6235** {'figure': 357, 'heading': 487, 'table': 1190, 'text': 4201}

Failed pages: `as_p0100`, `as_p0243`, `as_p0255`, `as_p0295`, `as_p1028`, `as_p0003`, `as_p0027`, `as_p0028`, `as_p0032`, `as_p0033`, `as_p0034`, `as_p0036`, `as_p0037`, `as_p0038`, `as_p0039`, `as_p0040`, `as_p0041`, `as_p0042`, `as_p0044`, `as_p0046`, `as_p0048`, `as_p0050`, `as_p0051`, `as_p0052`, `as_p0053`, `as_p0054`, `as_p0055`, `as_p0057`, `as_p0058`, `as_p0060`, `as_p0061`, `as_p0066`, `as_p0075`, `as_p0078`, `as_p0080`, `as_p0086`, `as_p0090`, `as_p0093`, `as_p0097`, `as_p0098`, `as_p0099`, `as_p0101`, `as_p0102`, `as_p0104`, `as_p0106`, `as_p0109`, `as_p0110`, `as_p0111`, `as_p0112`, `as_p0113`, `as_p0114`, `as_p0115`, `as_p0116`, `as_p0118`, `as_p0120`, `as_p0121`, `as_p0122`, `as_p0124`, `as_p0126`, `as_p0128`, `as_p0130`, `as_p0132`, `as_p0134`, `as_p0136`, `as_p0137`, `as_p0138`, `as_p0139`, `as_p0140`, `as_p0141`, `as_p0142`, `as_p0144`, `as_p0145`, `as_p0146`, `as_p0148`, `as_p0150`, `as_p0152`, `as_p0154`, `as_p0156`, `as_p0157`, `as_p0158`, `as_p0160`, `as_p0162`, `as_p0164`, `as_p0166`, `as_p0168`, `as_p0169`, `as_p0170`, `as_p0172`, `as_p0173`, `as_p0175`, `as_p0176`, `as_p0179`, `as_p0180`, `as_p0182`, `as_p0183`, `as_p0184`, `as_p0185`, `as_p0186`, `as_p0188`, `as_p0189`, `as_p0190`, `as_p0191`, `as_p0192`, `as_p0194`, `as_p0195`, `as_p0196`, `as_p0198`, `as_p0199`, `as_p0200`, `as_p0201`, `as_p0202`, `as_p0203`, `as_p0204`, `as_p0205`, `as_p0206`, `as_p0208`, `as_p0209`, `as_p0210`, `as_p0211`, `as_p0212`, `as_p0213`, `as_p0214`, `as_p0216`, `as_p0218`, `as_p0219`, `as_p0220`, `as_p0221`, `as_p0222`, `as_p0228`, `as_p0238`, `as_p0240`, `as_p0242`, `as_p0244`, `as_p0245`, `as_p0246`, `as_p0248`, `as_p0249`, `as_p0251`, `as_p0257`, `as_p0259`, `as_p0263`, `as_p0268`, `as_p0269`, `as_p0270`, `as_p0272`, `as_p0277`, `as_p0278`, `as_p0279`, `as_p0280`, `as_p0281`, `as_p0282`, `as_p0283`, `as_p0284`, `as_p0286`, `as_p0287`, `as_p0291`, `as_p0294`, `as_p0310`, `as_p0312`, `as_p0315`, `as_p0318`, `as_p0321`, `as_p0322`, `as_p0324`, `as_p0327`, `as_p0329`, `as_p0330`, `as_p0332`, `as_p0333`, `as_p0334`, `as_p0336`, `as_p0340`, `as_p0342`, `as_p0343`, `as_p0344`, `as_p0346`, `as_p0348`, `as_p0352`, `as_p0354`, `as_p0358`, `as_p0362`, `as_p0366`, `as_p0368`, `as_p0371`, `as_p0375`, `as_p0377`, `as_p0387`, `as_p0390`, `as_p0393`, `as_p0394`, `as_p0395`, `as_p0396`, `as_p0400`, `as_p0401`, `as_p0402`, `as_p0404`, `as_p0406`, `as_p0411`, `as_p0412`, `as_p0416`, `as_p0418`, `as_p0420`, `as_p0422`, `as_p0428`, `as_p0429`, `as_p0430`, `as_p0431`, `as_p0432`, `as_p0434`, `as_p0439`, `as_p0441`, `as_p0448`, `as_p0449`, `as_p0450`, `as_p0452`, `as_p0457`, `as_p0459`, `as_p0461`, `as_p0462`, `as_p0464`, `as_p0465`, `as_p0466`, `as_p0467`, `as_p0468`, `as_p0469`, `as_p0470`, `as_p0471`, `as_p0472`, `as_p0473`, `as_p0474`, `as_p0475`, `as_p0476`, `as_p0478`, `as_p0479`, `as_p0480`, `as_p0492`, `as_p0494`, `as_p0496`, `as_p0497`, `as_p0516`, `as_p0517`, `as_p0518`, `as_p0519`, `as_p0520`, `as_p0522`, `as_p0524`, `as_p0526`, `as_p0528`, `as_p0530`, `as_p0532`, `as_p0534`, `as_p0536`, `as_p0539`, `as_p0541`, `as_p0543`, `as_p0546`, `as_p0553`, `as_p0554`, `as_p0559`, `as_p0572`, `as_p0573`, `as_p0577`, `as_p0583`, `as_p0584`, `as_p0585`, `as_p0586`, `as_p0591`, `as_p0596`, `as_p0603`, `as_p0610`, `as_p0612`, `as_p0614`, `as_p0615`, `as_p0616`, `as_p0618`, `as_p0621`, `as_p0622`, `as_p0623`, `as_p0624`, `as_p0625`, `as_p0635`, `as_p0637`, `as_p0638`, `as_p0639`, `as_p0642`, `as_p0643`, `as_p0644`, `as_p0645`, `as_p0646`, `as_p0647`, `as_p0648`, `as_p0653`, `as_p0655`, `as_p0656`, `as_p0659`, `as_p0660`, `as_p0662`, `as_p0667`, `as_p0672`, `as_p0673`, `as_p0675`, `as_p0676`, `as_p0677`, `as_p0678`, `as_p0680`, `as_p0682`, `as_p0683`, `as_p0684`, `as_p0688`, `as_p0690`, `as_p0691`, `as_p0696`, `as_p0697`, `as_p0702`, `as_p0703`, `as_p0704`, `as_p0707`, `as_p0711`, `as_p0713`, `as_p0714`, `as_p0715`, `as_p0716`, `as_p0718`, `as_p0719`, `as_p0725`, `as_p0726`, `as_p0729`, `as_p0733`, `as_p0735`, `as_p0737`, `as_p0744`, `as_p0748`, `as_p0753`, `as_p0755`, `as_p0758`, `as_p0760`, `as_p0761`, `as_p0762`, `as_p0763`, `as_p0764`, `as_p0765`, `as_p0766`, `as_p0772`, `as_p0774`, `as_p0775`, `as_p0777`, `as_p0785`, `as_p0786`, `as_p0788`, `as_p0789`, `as_p0793`, `as_p0794`, `as_p0795`, `as_p0798`, `as_p0799`, `as_p0801`, `as_p0804`, `as_p0808`, `as_p0809`, `as_p0812`, `as_p0814`, `as_p0816`, `as_p0818`, `as_p0819`, `as_p0820`, `as_p0827`, `as_p0828`, `as_p0829`, `as_p0830`, `as_p0836`, `as_p0838`, `as_p0840`, `as_p0841`, `as_p0842`, `as_p0843`, `as_p0845`, `as_p0847`, `as_p0851`, `as_p0853`, `as_p0857`, `as_p0858`, `as_p0859`, `as_p0862`, `as_p0871`, `as_p0873`, `as_p0874`, `as_p0880`, `as_p0884`, `as_p0885`, `as_p0900`, `as_p0902`, `as_p0903`, `as_p0904`, `as_p0905`, `as_p0906`, `as_p0907`, `as_p0908`, `as_p0910`, `as_p0914`, `as_p0915`, `as_p0916`, `as_p0920`, `as_p0922`, `as_p0924`, `as_p0929`, `as_p0930`, `as_p0933`, `as_p0936`, `as_p0937`, `as_p0943`, `as_p0944`, `as_p0947`, `as_p0956`, `as_p0966`, `as_p0968`, `as_p0969`, `as_p0970`, `as_p0973`, `as_p0976`, `as_p0983`, `as_p0984`, `as_p0985`, `as_p0990`, `as_p0994`, `as_p0996`, `as_p1002`, `as_p1008`, `as_p1009`, `as_p1018`, `as_p1023`, `as_p1025`, `as_p1026`, `as_p1027`, `as_p1033`, `as_p1035`, `as_p1037`, `as_p1039`, `as_p1040`, `as_p1041`, `as_p1042`, `as_p1043`, `as_p1044`, `as_p1045`, `as_p1047`, `as_p1048`, `as_p1049`

## Timings

- corpus_render: 20.5 min
- loader: 0.9 min
- preprocess: 7.5 min
- layout: 4.4 min
- ocr_gated_subset: 5.2 min
- ocr_full_book: 277.5 min
- total: 316.9 min
