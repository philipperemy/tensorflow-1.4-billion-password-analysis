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

## Get started (processing + deep learning)

Process the data and run the first deep learning model:

```
# make sure to install the python deps first. Virtual env are recommended here.
# virtualenv -p python3 venv3; source venv3/bin/activate; pip install -r requirements.txt
# Remove "--max_num_files 100" to process the whole dataset (few hours and 20GB of free disk space are required.)
./process_and_train.sh <BreachCompilation path>
```

## Data (explanation)

```
INPUT:   BreachCompilation/
         BreachCompilation is organized as:

         - a/          - folder of emails starting with a
         - a/a         - file of emails starting with aa
         - a/b
         - a/d
         - ...
         - z/
         - ...
         - z/y
         - z/z

OUTPUT: - BreachCompilationAnalysis/edit-distance/1.csv
        - BreachCompilationAnalysis/edit-distance/2.csv
        - BreachCompilationAnalysis/edit-distance/3.csv
        [...]
        > cat 1.csv
            1 ||| samsung94 ||| samsung94@
            1 ||| 040384alexej ||| 040384alexey
            1 ||| HoiHalloDoeii14 ||| hoiHalloDoeii14
            1 ||| hoiHalloDoeii14 ||| hoiHalloDoeii13
            1 ||| hoiHalloDoeii13 ||| HoiHalloDoeii13
            1 ||| 8znachnuu ||| 7znachnuu
        EXPLANATION: edit-distance/ contains the passwords pairs sorted by edit distances.
        1.csv contains all pairs with edit distance = 1 (exactly one addition, substitution or deletion).
        2.csv => edit distance = 2, and so on.

        - BreachCompilationAnalysis/reduce-passwords-on-similar-emails/99_per_user.json
        - BreachCompilationAnalysis/reduce-passwords-on-similar-emails/9j_per_user.json
        - BreachCompilationAnalysis/reduce-passwords-on-similar-emails/9a_per_user.json
        [...]
        > cat 96_per_user.json
        {
            "1.0": [
            {
                "edit_distance": [
                    0,
                    1
                ],
                "email": "96-000@mail.ru",
                "password": [
                    "090698d",
                    "090698D"
                ]
            },
        {
                "edit_distance": [
                    0,
                    1
                ],
                "email": "96-96.1996@mail.ru",
                "password": [
                    "5555555555q",
                    "5555555555Q"
                ]
         }
        EXPLANATION: reduce-passwords-on-similar-emails/ contains files sorted by the first 2 letters of
        the email address. For example 96-000@mail.ru will be located in 96_per_user.json
        Each file lists all the passwords grouped by user and by edit distance.
        For example, 96-000@mail.ru had 2 passwords: 090698d and 090698D. The edit distance between them is 1.
        The edit_distance and the password arrays are of the same length, hence, a first 0 in the edit distance array.
        Those files are useful to model how users change passwords over time.
        We can't recover which one was the first password, but a shortest hamiltonian path algorithm is run
        to detect the most probably password ordering for a user. For example:
        hello => hello1 => hell@1 => hell@11 is the shortest path.
        We assume that users are lazy by nature and that they prefer to change their password by the lowest number
        of characters.
```

Run the data processing alone:

```
python3 run_data_processing.py --breach_compilation_folder <BreachCompilation path> --output_folder ~/BreachCompilationAnalysis
```

If the dataset is too big for you, you can set `max_num_files` to something between 0 and 2000.
 
- Make sure you have enough free memory (8GB should be enough).
- It took 1h30m to run on a Intel(R) Core(TM) i7-6900K CPU @ 3.20GHz (on a single thread).
- Uncompressed output is 13G.
