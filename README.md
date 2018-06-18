# 1.4 Billion Text Credentials Analysis (NLP)

Using deep learning and NLP to analyze a large corpus of clear text passwords.

Objectives:
- Train a generative model.
- Understand how people change their passwords over time: hello123 -> h@llo123 -> h@llo!23.

Disclaimer: for research purposes only.

## In the press

- [1.4 Billion Clear Text Credentials Discovered in a Single Database](https://medium.com/4iqdelvedeep/1-4-billion-clear-text-credentials-discovered-in-a-single-database-3131d0a1ae14)
- [Collection of 1.4 Billion Plain-Text Leaked Passwords Found Circulating Online](https://thehackernews.com/2017/12/data-breach-password-list.html)
- [Archive of 1.4 BEEELLION credentials in clear text found in dark web archive](https://www.theregister.co.uk/2017/12/12/archive_of_14_beeelion_credentials_in_clear_text_found_in_dark_web_archive/)
- [Forbes](https://www.forbes.com/sites/leemathews/2017/12/11/billion-hacked-passwords-dark-web/#74a6cf4221f2)



## Get the data

- Download any Torrent client.
- Here is a magnet link you can find on Reddit:
  - magnet:?xt=urn:btih:7ffbcd8cee06aba2ce6561688cf68ce2addca0a3&dn=BreachCompilation&tr=udp%3A%2F%2Ftracker.openbittorrent.com%3A80&tr=udp%3A%2F%2Ftracker.leechers-paradise.org%3A6969&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Fglotorrents.pw%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337
- Checksum list is available here: [checklist.chk](checklist.chk)
- `./count_total.sh` in `BreachCompilation` should display something like 1,400,553,870 rows.

## Deep Learning

- Stay tuned!

### Map the password list for each email

Generate the JSON files containing emails <-> list of passwords. Output folder is `~/BreachCompilationAnalysis`.

```
python3 read.py --breach_compilation_folder ~/BreachCompilation
```
 
- Make sure you have enough free memory (8GB should be enough).
- It took 1h30m to run on a Intel(R) Core(TM) i7-6900K CPU @ 3.20GHz (on a single thread).
- Uncompressed output is 13G.

Output is of the form:

```
> less ReducePasswordsOnSimilarEmailsCallback-z-b.json # emails starting with zb.
{
    "zb-email1@yahoo.com": [
        "pass1",
        "pass2"
    ],
    "zb-email2@yahoo.com": [
        "pass1",
        "pass2",
        "pass3"
    ],
    [...]
}
```
