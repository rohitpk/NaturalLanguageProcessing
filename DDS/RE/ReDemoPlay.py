import re
#Extracting Years pattern

sample_data = """ In the year 1526, first battle of Panipat was fought.
            The second battle at the same place was fought in the year 1556.
            In 1761, it was last battle fought at Panipat.
        """

def extract_years_without_re(text):
    numbers = ['1','2','3','4','5','6','7','8','9','0']
    years = []
    year = ''

    for character in text:
        if character in numbers:
            year=year+character
        else:
            if year:
                years.append(year)
            year = ''

    print(years)
    return years


def extract_years_using_re(text):
    re_for_extracting_numbers = r'\d{4}'
    years = re.findall(re_for_extracting_numbers,text)
    print(years)



if __name__=='__main__':
    # extract_years_without_re(sample_data)
    extract_years_using_re(sample_data)