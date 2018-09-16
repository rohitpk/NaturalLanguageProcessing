import itertools
import nltk
import string

def sentence_tokenize(text):
    return nltk.sent_tokenize(text)

def word_tokenize(text):
    return nltk.word_tokenize(text)

def pos_tag_sentence_words(sentence_words):
    return nltk.pos_tag_sents(sentence_words)

def extract_candidate_chunks(text, grammar=r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'):

    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # tokenize, POS-tag, and chunk using regular expressions
    chunker = nltk.chunk.regexp.RegexpParser(grammar)

    # Tokenize paragraph to sentences
    tokenize_by_sentence = sentence_tokenize(text)

    #Tokenize sentences into words
    words = []
    for sentence in tokenize_by_sentence:
        sentence_words = word_tokenize(sentence)
        words.append(sentence_words)

    # Tag words with POS tags
    tagged_sents = pos_tag_sentence_words(words)


    #Store Raw Chunks
    all_chunks=[]

    #Iterate tagged sentence
    for tagged_sent in tagged_sents:
        #Parse for the specified grammar
        parsed_sent = chunker.parse(tagged_sent)
        #assign IOB tags
        conll_tags = nltk.chunk.tree2conlltags(parsed_sent)

        all_chunks.append(conll_tags)

    #Merge the iterables
    all_chunks= list(itertools.chain.from_iterable(all_chunks))


    # join constituent chunk words into a single chunked phrase
    candidates = [' '.join(word for word, pos, chunk in group).lower() for key, group in
                       itertools.groupby(all_chunks, lambda word_pos_chunk: word_pos_chunk[2] != 'O') if key]


    return [cand for cand in candidates
            if cand not in stop_words and not all(char in punct for char in cand)]



if __name__=="__main__":
    text = '''
    World War II (often abbreviated to WWII or WW2), also known as the Second World War, was a global war that lasted from 1939 to 1945. The vast majority of the world's countries—including all the great powers—eventually formed two opposing military alliances: the Allies and the Axis. It was the most global war in history; it directly involved more than 100 million people from over 30 countries. In a state of total war, the major participants threw their entire economic, industrial, and scientific capabilities behind the war effort, blurring the distinction between civilian and military resources. World War II was the deadliest conflict in human history, marked by 50 to 85 million fatalities, most of whom were civilians in the Soviet Union and China. It included massacres, the genocide of the Holocaust, strategic bombing, premeditated death from starvation and disease and the only use of nuclear weapons in war.[1][2][3][4]

The Empire of Japan aimed to dominate Asia and the Pacific and was already at war with the Republic of China in 1937,[5][b] but the world war is generally said to have begun on 1 September 1939,[6] the day of the invasion of Poland by Nazi Germany and subsequent declarations of war on Germany by France and the United Kingdom. From late 1939 to early 1941, in a series of campaigns and treaties, Germany conquered or controlled much of continental Europe, and formed the Axis alliance with Italy and Japan. Under the Molotov–Ribbentrop Pact of August 1939, Germany and the Soviet Union partitioned and annexed territories of their European neighbours, Poland, Finland, Romania and the Baltic states. The war continued primarily between the European Axis powers and the British Commonwealth, with campaigns in North Africa, East Africa, the Balkans, the aerial Battle of Britain, the Blitz bombing campaign, and the long Battle of the Atlantic. On 22 June 1941, the European Axis powers launched an invasion of the Soviet Union, opening the largest land theatre of war in history, which trapped the Axis, most crucially the German Wehrmacht, into a war of attrition. In December 1941, Japan attacked the United States and European colonies in the Pacific Ocean, and quickly conquered much of the Western Pacific. The European Axis powers quickly declared war on the United States of America in support of their Japanese ally. The Japanese conquests were perceived by many in Asia as liberation from Western dominance; as such, several armies from the conquered territories aided the Japanese.
    
    '''.encode('ascii','ignore')
    print(set(extract_candidate_chunks(str(text))))