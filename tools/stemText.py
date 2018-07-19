############################## PorterStemmer #############################

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string


def stem_text(title) :

    """
        title is a string not unicode string so it will give error UnicodeDecodeError
        So to fix this, first convert(Decode) title into unicode and then encode that unicode object to string.
    """
    # https://stackoverflow.com/questions/21129020/how-to-fix-unicodedecodeerror-ascii-codec-cant-decode-byte
    if isinstance(title, str):
        title = unicode(title, 'utf-8')

    if isinstance(title, str):
        #note: this removes the character and encodes back to string.
        title = title.decode('ascii', 'ignore').encode('ascii')
    elif isinstance(title, unicode):
        title = title.encode('ascii', 'ignore')


    """
        remove stop words(is, the, for etc) and perform stemming to reduce a word to its stem and
        return a string containing important information about title
    """

    ### convert title into lowercase
    title = title.lower()

    ### remove punctuation
    title = title.translate(string.maketrans("", ""), string.punctuation)

    ### split the text string into individual words
    word_data = title.split() # convert into list of individual words

    ### remove stop words
    sw = stopwords.words("english")
    new_word_data = []
    for word in word_data :
        if word not in sw :
            new_word_data.append(word)

    # this string will contain stem words of title
    new_title = ""

    porter = PorterStemmer()

    for word in new_word_data :
        new_title += porter.stem(word) + " "


    return new_title



def main() :
    print "You are running this module as a program"

if __name__ == '__main__':
    main()
