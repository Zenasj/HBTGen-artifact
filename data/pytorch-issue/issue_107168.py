models = [
    {
        "name": "bert-base-uncased",
        "input": [
            {"inputs": "Paris is [MASK] of France."},
            {"inputs": "Number theory (or arithmetic or higher arithmetic in older usage) is a branch of pure mathematics devoted primarily to the study of the integers and arithmetic functions. German mathematician Carl Friedrich Gauss (1777–1855) said, \"Mathematics is the queen of the sciences—and number theory is the queen of mathematics.\" Number theorists study [MASK] numbers as well as the properties of mathematical objects constructed from integers (for example, rational numbers), or defined as generalizations of the integers (for example, algebraic integers)."}
        ]
    },
    {
        "name": "emilyalsentzer/Bio_ClinicalBERT",
        "input": [
            {"inputs": "Paris is [MASK] of France."},
            {"inputs": "Number theory (or arithmetic or higher arithmetic in older usage) is a branch of pure mathematics devoted primarily to the study of the integers and arithmetic functions. German mathematician Carl Friedrich Gauss (1777–1855) said, \"Mathematics is the queen of the sciences—and number theory is the queen of mathematics.\" Number theorists study [MASK] numbers as well as the properties of mathematical objects constructed from integers (for example, rational numbers), or defined as generalizations of the integers (for example, algebraic integers)."}
        ]
    },
    {
        "name": "gpt2",
        "input": [
            {"text_inputs": "Paris is"},
            {"text_inputs": "My name is Julien and I like to keep my company. Like, I think it's fun, it keeps me company, and it also helps the guys with their health care. We got a good group of guys and ladies"}
        ]
    },
    {
        "name": "prajjwal1/bert-tiny",
        "input": [
            {"inputs": "Paris is [MASK] of France."},
            {"inputs": "Number theory (or arithmetic or higher arithmetic in older usage) is a branch of pure mathematics devoted primarily to the study of the integers and arithmetic functions. German mathematician Carl Friedrich Gauss (1777–1855) said, \"Mathematics is the queen of the sciences—and number theory is the queen of mathematics.\" Number theorists study [MASK] numbers as well as the properties of mathematical objects constructed from integers (for example, rational numbers), or defined as generalizations of the integers (for example, algebraic integers)."}
        ]
    },
    {
        "name": "xlm-roberta-base",
        "input": [
            {"inputs": "Paris is <mask> of France."},
            {"inputs": "Number theory (or arithmetic or higher arithmetic in older usage) is a branch of pure mathematics devoted primarily to the study of the integers and arithmetic functions. German mathematician Carl Friedrich Gauss (1777–1855) said, \"Mathematics is the queen of the sciences—and number theory is the queen of mathematics.\" Number theorists study <mask> numbers as well as the properties of mathematical objects constructed from integers (for example, rational numbers), or defined as generalizations of the integers (for example, algebraic integers)."}
        ]
    },
    {
        "name": "distilbert-base-uncased",
        "input": [
            {"inputs": "Paris is [MASK] of France."},
            {"inputs": "Number theory (or arithmetic or higher arithmetic in older usage) is a branch of pure mathematics devoted primarily to the study of the integers and arithmetic functions. German mathematician Carl Friedrich Gauss (1777–1855) said, \"Mathematics is the queen of the sciences—and number theory is the queen of mathematics.\" Number theorists study [MASK] numbers as well as the properties of mathematical objects constructed from integers (for example, rational numbers), or defined as generalizations of the integers (for example, algebraic integers)."}
        ]
    },
    {
        "name": "StanfordAIMI/stanford-deidentifier-base",
        "input": [
            {"inputs": "Paris is capital of France."},
            {"inputs": "PROCEDURE: Chest xray. COMPARISON: last seen on 1/1/2020 and also record dated of March 1st, 2019. FINDINGS: patchy airspace opacities. IMPRESSION: The results of the chest xray of January 1 2020 are the most concerning ones. The patient was transmitted to another service of UH Medical Center under the responsability of Dr. Perez. We used the system MedClinical data transmitter and sent the data on 2/1/2020, under the ID 5874233. We received the confirmation of Dr Perez. He is reachable at 567-493-1234."}
        ]
    },
    {
        "name": "xlm-roberta-large",
        "input": [
            {"inputs": "Paris is <mask> of France."},
            {"inputs": "Number theory (or arithmetic or higher arithmetic in older usage) is a branch of pure mathematics devoted primarily to the study of the integers and arithmetic functions. German mathematician Carl Friedrich Gauss (1777–1855) said, \"Mathematics is the queen of the sciences—and number theory is the queen of mathematics.\" Number theorists study <mask> numbers as well as the properties of mathematical objects constructed from integers (for example, rational numbers), or defined as generalizations of the integers (for example, algebraic integers)."}
        ]
    },
    {
        "name": "t5-base",
        "input": [
            ("My name is Wolfgang and I live in Berlin",),
            ("Number theory (or arithmetic or higher arithmetic in older usage) is a branch of pure mathematics devoted primarily to the study of the integers and arithmetic functions. German mathematician Carl Friedrich Gauss (1777–1855) said, \"Mathematics is the queen of the sciences—and number theory is the queen of mathematics.\" Number theorists study prime numbers as well as the properties of mathematical objects constructed from integers (for example, rational numbers), or defined as generalizations of the integers (for example, algebraic integers).",)
        ]
    },
    {
        "name": "bert-base-cased",
        "input": [
            {"inputs": "Paris is [MASK] of France."},
            {"inputs": "Number theory (or arithmetic or higher arithmetic in older usage) is a branch of pure mathematics devoted primarily to the study of the integers and arithmetic functions. German mathematician Carl Friedrich Gauss (1777–1855) said, \"Mathematics is the queen of the sciences—and number theory is the queen of mathematics.\" Number theorists study [MASK] numbers as well as the properties of mathematical objects constructed from integers (for example, rational numbers), or defined as generalizations of the integers (for example, algebraic integers)."}
        ]
    },
    {
        "name": "roberta-base",
        "input": [
            {"inputs": "Paris is <mask> of France."},
            {"inputs": "Number theory (or arithmetic or higher arithmetic in older usage) is a branch of pure mathematics devoted primarily to the study of the integers and arithmetic functions. German mathematician Carl Friedrich Gauss (1777–1855) said, \"Mathematics is the queen of the sciences—and number theory is the queen of mathematics.\" Number theorists study <mask> numbers as well as the properties of mathematical objects constructed from integers (for example, rational numbers), or defined as generalizations of the integers (for example, algebraic integers)."}
        ]
    },
    {
        "name": "albert-base-v2",
        "input": [
            {"inputs": "Paris is [MASK] of France."},
            {"inputs": "Number theory (or arithmetic or higher arithmetic in older usage) is a branch of pure mathematics devoted primarily to the study of the integers and arithmetic functions. German mathematician Carl Friedrich Gauss (1777–1855) said, \"Mathematics is the queen of the sciences—and number theory is the queen of mathematics.\" Number theorists study [MASK] numbers as well as the properties of mathematical objects constructed from integers (for example, rational numbers), or defined as generalizations of the integers (for example, algebraic integers)."}
        ]
    },
    {
        "name": "distilroberta-base",
        "input": [
            {"inputs": "Paris is <mask> of France."},
            {"inputs": "Number theory (or arithmetic or higher arithmetic in older usage) is a branch of pure mathematics devoted primarily to the study of the integers and arithmetic functions. German mathematician Carl Friedrich Gauss (1777–1855) said, \"Mathematics is the queen of the sciences—and number theory is the queen of mathematics.\" Number theorists study <mask> numbers as well as the properties of mathematical objects constructed from integers (for example, rational numbers), or defined as generalizations of the integers (for example, algebraic integers)."}
        ]
    },
    {
        "name": "distilbert-base-uncased-finetuned-sst-2-english",
        "input": [
            ("Paris is capital of France.",),
            ("PROCEDURE: Chest xray. COMPARISON: last seen on 1/1/2020 and also record dated of March 1st, 2019. FINDINGS: patchy airspace opacities. IMPRESSION: The results of the chest xray of January 1 2020 are the most concerning ones. The patient was transmitted to another service of UH Medical Center under the responsability of Dr. Perez. We used the system MedClinical data transmitter and sent the data on 2/1/2020, under the ID 5874233. We received the confirmation of Dr Perez. He is reachable at 567-493-1234.",)
        ]
    },
    {
        "name": "google/electra-base-discriminator",
        "input": [
            {"inputs": "Paris is [MASK] of France."},
            {"inputs": "Number theory (or arithmetic or higher arithmetic in older usage) is a branch of pure mathematics devoted primarily to the study of the integers and arithmetic functions. German mathematician Carl Friedrich Gauss (1777–1855) said, \"Mathematics is the queen of the sciences—and number theory is the queen of mathematics.\" Number theorists study [MASK] numbers as well as the properties of mathematical objects constructed from integers (for example, rational numbers), or defined as generalizations of the integers (for example, algebraic integers)."}
        ]
    },
    {
        "name": "facebook/bart-large-mnli",
        "input": [
            {"sequences": "I have a problem with my iphone that needs to be resolved asap!!", "candidate_labels": ["urgent", "not urgent", "phone", "tablet", "computer"]},
            {"sequences": "A new model offers an explanation for how the Galilean satellites formed around the solar system’s largest world. Konstantin Batygin did not set out to solve one of the solar system’s most puzzling mysteries when he went for a run up a hill in Nice, France. Dr. Batygin, a Caltech researcher, best known for his contributions to the search for the solar system’s missing “Planet Nine,” spotted a beer bottle. At a steep, 20 degree grade, he wondered why it wasn’t rolling down the hill. He realized there was a breeze at his back holding the bottle in place. Then he had a thought that would only pop into the mind of a theoretical astrophysicist: “Oh! This is how Europa formed.” Europa is one of Jupiter’s four large Galilean moons. And in a paper published Monday in the Astrophysical Journal, Dr. Batygin and a co-author, Alessandro Morbidelli, a planetary scientist at the Côte d’Azur Observatory in France, present a theory explaining how some moons form around gas giants like Jupiter and Saturn, suggesting that millimeter-sized grains of hail produced during the solar system’s formation became trapped around these massive worlds, taking shape one at a time into the potentially habitable moons we know today.", "candidate_labels": ["space & cosmos", "scientific discovery", "microbiology", "robots", "archeology"]}
        ]
    },
    {
        "name": "bert-base-multilingual-cased",
        "input": [
            {"inputs": "Paris is [MASK] of France."},
            {"inputs": "Number theory (or arithmetic or higher arithmetic in older usage) is a branch of pure mathematics devoted primarily to the study of the integers and arithmetic functions. German mathematician Carl Friedrich Gauss (1777–1855) said, \"Mathematics is the queen of the sciences—and number theory is the queen of mathematics.\" Number theorists study [MASK] numbers as well as the properties of mathematical objects constructed from integers (for example, rational numbers), or defined as generalizations of the integers (for example, algebraic integers)."}
        ]
    },
    {
        "name": "philschmid/bart-large-cnn-samsum",
        "input": [
            ("Paris is capital of France.",),
            ("PROCEDURE: Chest xray. COMPARISON: last seen on 1/1/2020 and also record dated of March 1st, 2019. FINDINGS: patchy airspace opacities. IMPRESSION: The results of the chest xray of January 1 2020 are the most concerning ones. The patient was transmitted to another service of UH Medical Center under the responsability of Dr. Perez. We used the system MedClinical data transmitter and sent the data on 2/1/2020, under the ID 5874233. We received the confirmation of Dr Perez. He is reachable at 567-493-1234.",)
        ]
    },
    {
        "name": "bert-large-uncased",
        "input": [
            {"inputs": "Paris is [MASK] of France."},
            {"inputs": "Number theory (or arithmetic or higher arithmetic in older usage) is a branch of pure mathematics devoted primarily to the study of the integers and arithmetic functions. German mathematician Carl Friedrich Gauss (1777–1855) said, \"Mathematics is the queen of the sciences—and number theory is the queen of mathematics.\" Number theorists study [MASK] numbers as well as the properties of mathematical objects constructed from integers (for example, rational numbers), or defined as generalizations of the integers (for example, algebraic integers)."}
        ]
    },
    {
        "name": "cl-tohoku/bert-base-japanese-whole-word-masking",
        "input": [
            {"inputs": "Paris is [MASK] of France."},
            {"inputs": "Number theory (or arithmetic or higher arithmetic in older usage) is a branch of pure mathematics devoted primarily to the study of the integers and arithmetic functions. German mathematician Carl Friedrich Gauss (1777–1855) said, \"Mathematics is the queen of the sciences—and number theory is the queen of mathematics.\" Number theorists study [MASK] numbers as well as the properties of mathematical objects constructed from integers (for example, rational numbers), or defined as generalizations of the integers (for example, algebraic integers)."}
        ]
    },
    {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "input": [
            {"inputs": "Paris is [MASK] of France."},
            {"inputs": "Number theory (or arithmetic or higher arithmetic in older usage) is a branch of pure mathematics devoted primarily to the study of the integers and arithmetic functions. German mathematician Carl Friedrich Gauss (1777–1855) said, \"Mathematics is the queen of the sciences—and number theory is the queen of mathematics.\" Number theorists study [MASK] numbers as well as the properties of mathematical objects constructed from integers (for example, rational numbers), or defined as generalizations of the integers (for example, algebraic integers)."}
        ]
    },
    {
        "name": "Helsinki-NLP/opus-mt-en-es",
        "input": [
            ("My name is Wolfgang and I live in Berlin",),
            ("Number theory (or arithmetic or higher arithmetic in older usage) is a branch of pure mathematics devoted primarily to the study of the integers and arithmetic functions. German mathematician Carl Friedrich Gauss (1777–1855) said, \"Mathematics is the queen of the sciences—and number theory is the queen of mathematics.\" Number theorists study prime numbers as well as the properties of mathematical objects constructed from integers (for example, rational numbers), or defined as generalizations of the integers (for example, algebraic integers).",)
        ]
    },
    {
        "name": "t5-small",
        "input": [
            ("My name is Wolfgang and I live in Berlin",),
            ("Number theory (or arithmetic or higher arithmetic in older usage) is a branch of pure mathematics devoted primarily to the study of the integers and arithmetic functions. German mathematician Carl Friedrich Gauss (1777–1855) said, \"Mathematics is the queen of the sciences—and number theory is the queen of mathematics.\" Number theorists study prime numbers as well as the properties of mathematical objects constructed from integers (for example, rational numbers), or defined as generalizations of the integers (for example, algebraic integers).",)
        ]
    },
]