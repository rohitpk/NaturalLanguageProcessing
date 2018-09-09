
import re

#Extracting Years pattern
sample_data = """ In the year 1526, first battle of Panipat was fought.
            The second battle at the same place was fought in the year 1556.
            In 1761, it was last battle fought at Panipat.
        """

def get_years_without_re(text):

    number_list = ['0','1','2','3','4','5','6','7','8','9']
    years = []
    year = ''
    for character in text:
        if character in number_list:
            year=year+character
        else:
            if year:
                years.append(year)
            year = ''
    return years


def get_years_with_re(text):
    regular_expression = r'\d{4}'
    years = re.findall(regular_expression,text)
    return years

if __name__=='__main__':
    years_without_re = get_years_without_re(sample_data)
    print(">>>>>>years_without_re\n",years_without_re)

    years_with_re = get_years_with_re(sample_data)
    print(">>>>>>years_with_re\n", years_with_re)


