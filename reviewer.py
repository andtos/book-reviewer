from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import sys



def main():
    args = sys.argv
    if len(args) == 2:
        parse(args[1])
    else:
        print("Include a link to bookmeter")


def parse(link):
    driver = webdriver.Chrome() 
    driver.get(link)
    try:    
        div = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'section.bm-page-loader[data-status=idle]')))
        spans = div.find_elements(By.TAG_NAME, 'span')
        with open('reviews.txt', 'w', encoding='utf-8') as txtfile:
            for span in spans:
                text = span.text
                if text != '' and text != 'ネタバレ' and text != 'ナイス' and text.isdigit() == False:
                    txtfile.write(text)
                    txtfile.write("\n")
    finally:
        driver.quit()




if __name__=="__main__":
    main()