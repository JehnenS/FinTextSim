import random
import re
from tqdm import tqdm

class Masker:
    """
    Mask train set to avoid overrelying on easy keyword heuristics --> learn the semantics instead
    """
    def __init__(
        self,
        labeled_dataset,
        label_to_keywords,
        tokenizer,
        label_to_blacklist=None,
        #mask_prob=0.7,
        #use_mask_token=False
    ):
        self.data = labeled_dataset
        self.tokenizer = tokenizer
        #self.mask_prob = mask_prob
        #self.use_mask_token = use_mask_token

        #clean the label_to_keywords
        self.label_to_keywords = {
            lbl: [k.lower() for k in kws]
            for lbl, kws in label_to_keywords.items()
        }

        #clean the label_to_blacklist
        self.label_to_blacklist = {
            lbl: [b.lower() for b in bls]
            for lbl, bls in (label_to_blacklist or {}).items()
        }

        #set up cleaning pattern
        self._clean_re = re.compile(r"[^\w]")

    def _normalize(self, word):
        """
        clean the sentences
        """
        return self._clean_re.sub("", word.lower())

    def mask_sentence(self, text, label, mask_prob:float = 0.7, use_mask_token:bool = False):
        """
        mask each sentence
        """
        #determine keywords and blacklist for this label
        keywords = self.label_to_keywords.get(label, [])
        blacklist = self.label_to_blacklist.get(label, [])

        #split sentences into words
        words = text.split()

        #create list to store results
        masked_words = []

        #iterate over all words in the sentence
        for word in words:
            clean = self._normalize(word) #clean the sentence
            has_keyword = any(k in clean for k in keywords) #check if keyword-substring is existent
            has_blacklist = any(b in clean for b in blacklist) #check if it is not in the blacklist

            #if keyword is a substring of word and word is not in blacklist, apply random masking
            if has_keyword and not has_blacklist and random.random() < mask_prob:
                if use_mask_token:
                    masked_words.append(self.tokenizer.mask_token)  # [MASK] --> use MASK token in text
                # else: skip the word (removal)
            else:
                masked_words.append(word) #

        return " ".join(masked_words)

    def get_masked_data(self, mask_prob:float = 0.7, use_mask_token:bool = False):
        """
        output the data in the required tuple format of text, label
        """
        return [(self.mask_sentence(text, label, mask_prob, use_mask_token), label) for text, label in tqdm(self.data, desc = "Masking progress")]
