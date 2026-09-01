"""Fixed evaluation set for the RAG pipeline (committed = reproducible).

Ground-truth Q/A over data/the-metamorphosis.pdf (Kafka). Keep this FIXED across
retrieval experiments so RAGAS deltas are attributable to the system change, not
the questions. ~15 items so scores aren't dominated by n=4 noise.
"""

EVAL_SET = [
    {"question": "What does Gregor Samsa wake up transformed into?",
     "reference": "A giant insect / monstrous vermin."},
    {"question": "What is Gregor Samsa's profession?",
     "reference": "He is a traveling salesman."},
    {"question": "Why does Gregor keep working a job he dislikes?",
     "reference": "To pay off his parents' debt to his employer."},
    {"question": "Who is Grete?",
     "reference": "Gregor's younger sister."},
    {"question": "Who first takes on the task of feeding Gregor after his transformation?",
     "reference": "His sister, Grete."},
    {"question": "What kind of food does Gregor come to prefer after changing?",
     "reference": "Rotten or spoiled leftovers rather than fresh food."},
    {"question": "Who visits early on to demand why Gregor missed work?",
     "reference": "The chief clerk (office manager) from his firm."},
    {"question": "What does Gregor's father throw at him, wounding him?",
     "reference": "Apples; one lodges in his back and injures him."},
    {"question": "How does the family support itself once Gregor can no longer work?",
     "reference": "Family members take jobs and rent a room to lodgers/boarders."},
    {"question": "What instrument does Grete play for the lodgers?",
     "reference": "The violin."},
    {"question": "What does Grete finally argue the family must do about Gregor?",
     "reference": "Get rid of it; stop believing the creature is still Gregor."},
    {"question": "How does Gregor die?",
     "reference": "Weak, injured, and neglected, he dies alone in his room."},
    {"question": "Who discovers Gregor's dead body?",
     "reference": "The charwoman (cleaning lady)."},
    {"question": "How does the family react after Gregor's death?",
     "reference": "With relief; they take the day off and go on an outing to the countryside."},
    {"question": "What do Gregor's parents notice about Grete at the very end?",
     "reference": "That she has grown into an attractive young woman ready for marriage."},
]
