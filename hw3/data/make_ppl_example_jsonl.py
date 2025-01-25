import json

cmu_wiki = """The Carnegie Technical Schools were founded in 1900 in Pittsburgh, Pennsylvania[19] by the Scottish-American industrialist and philanthropist Andrew Carnegie, who wrote "My heart is in the work", when he donated the funds to create the institution. Carnegie's vision was to open a vocational training school for the sons and daughters of working-class Pittsburghers, many of whom worked in his mills. Carnegie was inspired for the design of his school by the Pratt Institute in Brooklyn, New York, founded by industrialist Charles Pratt in 1887.[20] In 1912, the institution changed its name to Carnegie Institute of Technology (CIT) and began offering four-year degrees. During this time, CIT consisted of four constituent schools: the School of Fine and Applied Arts, the School of Apprentices and Journeymen, the School of Science and Technology, and the Margaret Morrison Carnegie School for Women.

The Mellon Institute of Industrial Research was founded in 1913 by banker and industrialist brothers Andrew Mellon (who went on to become U.S. Treasury Secretary) and Richard B. Mellon in honor of their father, Thomas Mellon, patriarch of the Mellon family. The Institute began as a research organization that performed contract work for government and industry, initially as a department within the University of Pittsburgh. In 1927, the Mellon Institute was incorporated as an independent nonprofit. In 1937, the Mellon Institute's iconic building was completed on Fifth Avenue.[21]

In 1967, with support from Paul Mellon, the Carnegie Institute of Technology merged with the Mellon Institute of Industrial Research to become Carnegie Mellon University. In 1973, Carnegie Mellon's coordinate women's college, the Margaret Morrison Carnegie College, merged its academic programs with the rest of the university.[22] The industrial research mission of the Mellon Institute survived the merger as the Carnegie Mellon Research Institute (CMRI) and continued doing work on contract to industry and government. In 2001, CMRI's programs were subsumed by other parts of the university or spun off into autonomous entities.
"""

jabberwocky = """’Twas brillig, and the slithy toves
      Did gyre and gimble in the wabe:
All mimsy were the borogoves,
      And the mome raths outgrabe.

“Beware the Jabberwock, my son!
      The jaws that bite, the claws that catch!
Beware the Jubjub bird, and shun
      The frumious Bandersnatch!”

He took his vorpal sword in hand;
      Long time the manxome foe he sought—
So rested he by the Tumtum tree
      And stood awhile in thought.

And, as in uffish thought he stood,
      The Jabberwock, with eyes of flame,
Came whiffling through the tulgey wood,
      And burbled as it came!

One, two! One, two! And through and through
      The vorpal blade went snicker-snack!
He left it dead, and with its head
      He went galumphing back.

“And hast thou slain the Jabberwock?
      Come to my arms, my beamish boy!
O frabjous day! Callooh! Callay!”
      He chortled in his joy.

’Twas brillig, and the slithy toves
      Did gyre and gimble in the wabe:
All mimsy were the borogoves,
      And the mome raths outgrabe."""

examples = [cmu_wiki, jabberwocky]

with open("ppl_eval_examples.jsonl", "w") as f:
  for example in examples:
    f.write(json.dumps({"document": example}) + "\n")