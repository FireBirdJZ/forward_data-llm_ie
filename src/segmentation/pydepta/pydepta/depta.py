# from urllib3 import Request, urlopen
# from scrapely import HtmlPage, Scraper, TemplateMaker, best_match, InstanceBasedLearningExtractor
# from lxml.html import tostring
# from scrapely.extraction.regionextract import (RecordExtractor, BasicTypeExtractor, TemplatePageExtractor, \
#                                                TraceExtractor, labelled_element, attrs2dict)
# from scrapely.extraction.similarity import first_longest_subsequence
# from scrapely.template import FragmentNotFound, FragmentAlreadyAnnotated
# from w3lib.encoding import html_to_unicode

from htmls import DomTreeBuilder
from mdr import MiningDataRegion, MiningDataRecord, MiningDataField
import lxml
import requests
from playwright.sync_api import sync_playwright

from extract_prof_names import ProfessorExtractor
from comparing_models import FacultyDataHarvester

# class DeptaTemplateMaker(TemplateMaker):

#     def annotate(self, field, score_func, best_match=False):
#         """Annotate a field, but unlike TemplateMaker, it simply try next match field if need.
#         """
#         indexes = self.select(score_func)
#         if not indexes:
#             raise FragmentNotFound("Fragment not found annotating %r using: %s" %
#                                    (field, score_func))
#         if best_match:
#             del indexes[1:]
#         for i in indexes:
#             try:
#                 if self.annotate_fragment(i, field):
#                     break
#             except FragmentAlreadyAnnotated:
#                 pass

# class DeptaExtractor(RecordExtractor):
#     """
#     A simple RecordExtractor variant to handle the tabulated data.
#     """
#     def __init__(self, extractors, template_tokens):
#         super(DeptaExtractor, self).__init__(extractors, template_tokens)
#         self.best_match = first_longest_subsequence

#     def extract(self, page, start_index=0, end_index=None, ignored_regions=None, **kwargs):
#         if ignored_regions is None:
#             ignored_regions = []
#         region_elements = sorted(self.extractors + ignored_regions, key=lambda x: labelled_element(x).start_index)
#         pindex, sindex, attributes = self._doextract(page, region_elements, start_index,
#                                            end_index, **kwargs)

#         if not end_index:
#             end_index = len(page.page_tokens)

#         # collect variant data, maintaining the order of variants
#         r = []
#         items = [(k, v) for k, v in attributes]

#         # if the number of extracted data match
#         if len(items) == len(region_elements):
#             r.append(attrs2dict(items))

#             # if there are remaining items, extract recursively backward
#             if sindex and sindex < end_index:
#                 r.extend(self.extract(page, 0, pindex - 1, ignored_regions, **kwargs))
#         return r

#     def __repr__(self):
#         return str(self)

# class DeptaIBLExtractor(InstanceBasedLearningExtractor):

#     def build_extraction_tree(self, template, type_descriptor, trace=True):
#         """Build a tree of region extractors corresponding to the
#         template
#         """
#         attribute_map = type_descriptor.attribute_map if type_descriptor else None
#         extractors = BasicTypeExtractor.create(template.annotations, attribute_map)
#         if trace:
#             extractors = TraceExtractor.apply(template, extractors)
#         for cls in (DeptaExtractor,):
#             extractors = cls.apply(template, extractors)
#             if trace:
#                 extractors = TraceExtractor.apply(template, extractors)

#         return TemplatePageExtractor(template, extractors)

#     def extract(self, html, pref_template_id=None):
#         extracted, template = super(DeptaIBLExtractor, self).extract(html, pref_template_id)
#         if extracted:
#             return extracted[::-1], template
#         return None, None

class Depta(object):
    def __init__(self, threshold=0.75, k=5):
        self.threshold = threshold
        self.k = k
        # self.scraper = Scraper()

    def extract(self, html='', **kwargs):
        """
        extract data field from raw html or from a url.
        """
        # if not html and 'url' in kwargs:
        #     req = Request(kwargs.pop('url'))
        #     req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:98.0) Gecko/20100101 Firefox/98.0')
        #     info = urlopen(req)
        #     _, html = html_to_unicode(info.headers.get('content_type'), info.read())

        builder = DomTreeBuilder(html)
        root = builder.build()

        region_finder = MiningDataRegion(root, self.k, self.threshold)
        #print(f"region_finder: {region_finder}")
        regions = region_finder.find_regions(root)
        #print(f"regions: {regions}")

        record_finder = MiningDataRecord(self.threshold)
        field_finder = MiningDataField()

        for region in regions:
            records = record_finder.find_records(region)
            items, _ = field_finder.align_records(records)
            region.items = items
            region.records = records
            # if 'verbose' in kwargs:
            #     print(region)
            #     for record in records:
            #         print('\t', record)

        return regions

    # def train(self, seed, data):
    #     """
    #     train scrapely from give seed region and data.
    #     """
    #     assert data, "Cannot train with empty data"
    #     htmlpage = self._region_to_htmlpage(seed)
    #     dtm = DeptaTemplateMaker(htmlpage)
    #     if isinstance(data, dict):
    #         data = data.items()

    #     for field, values in data:
    #         if not hasattr(values, '__iter__'):
    #             values = [values]
    #         for value in values:
    #             if isinstance(value, str):
    #                 value = value.decode(htmlpage.encoding or 'utf-8')
    #             dtm.annotate(field, best_match(value))
    #     self.scraper.add_template(dtm.get_template())


    # def infer(self, html='', **kwargs):
    #     """
    #     extract data with seed region and the data you expect to scrape from there.
    #     """
    #     # if 'url' in kwargs:
    #     #     info = urlopen(kwargs.pop('url'))
    #     #     _, html = html_to_unicode(info.headers.get('content_type'), info.read())

    #     builder = DomTreeBuilder(html)
    #     doc = builder.build()
    #     page = HtmlPage(body=tostring(doc, encoding='utf-8', method='html'))

    #     return self._scrape_page(page)

    # def _scrape_page(self, page):
    #     if self.scraper._ex is None:
    #         self.scraper._ex = DeptaIBLExtractor((t, None) for t in
    #             self.scraper._templates)
    #     return self.scraper._ex.extract(page)[0]

    # def _region_to_htmlpage(self, region):
    #     seed_body = tostring(region.parent[region.start], encoding='utf-8', method='html')
    #     return HtmlPage(body=seed_body)

def fetch_html_from_url(url):
    raw_html = ""

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        
        # Function to handle the response
        def handle_response(response):
            nonlocal raw_html
            if response.url == url and response.status == 200:
                raw_html = response.text()

        # Listen to the 'response' event
        page.on("response", handle_response)

        page.goto(url)
        browser.close()

    return raw_html


# class Test():
#     def __init__(self) -> None:
#         pass




def main():
    d = Depta()
    url = 'https://cs.illinois.edu/about/people/all-faculty'
    #url = 'https://www.eecs.mit.edu/role/faculty-cs/'
    #url = 'https://inf.ethz.ch/people/faculty/faculty.html'
    #url = "https://csd.cmu.edu/people/faculty"
    #url = 'https://cse.engin.umich.edu/people/faculty/'
    #url = 'https://www.cs.stanford.edu/people/faculty'
    #url = 'https://www.cs.princeton.edu/people/faculty'
    # Shopify page can cause depta to crash
    #url = "https://www.shopify.com/blog/ecommerce-seo-beginners-guide" #Returns empty Doc if full html is used, works for ex: [2000:100000ß]
    # Fetch the HTML content from the URL
    #response = requests.get(url)

    # html_content = fetch_html_from_url(url)
    # print(html_content)
    # professor_extractor = ProfessorExtractor()
    # html_content = professor_extractor.fetch_html_from_url(url)
    # text_content = professor_extractor.extract_text()
    #print(html_content)
    #print(text_content)

    # Actual Run
    # for region in regions:
    #     print('------------------------------------------------------------------------------------------')
    #     #print(region.as_plain_texts())
    #     #professor_extractor.find_names_in_region(region.as_plain_texts())
    #     for record in region.as_plain_texts():
    #     #print(region.as_html_table())
    #         print(record)
    # # Now Take set of names and put them in json file
    # #professor_extractor.extract_prof_names_to_json()



 

  

    ## Comparing models
    faculty_data_harvester = FacultyDataHarvester()
    #html_content = faculty_data_harvester.fetch_html_from_url(url)
    html_content = faculty_data_harvester.load_html_from_file(url)
    text_content = faculty_data_harvester.extract_text()

    # Extract data regions using Depta
    regions: list = d.extract(html=html_content)
    #print(regions)
    #Print the extracted data
    #### For DEBUGGING
    #i = 0
    for region in regions:
        print('------------------------------------------------------------------------------------------')
        # i+=1
        # if i <= 20: 
        #     continue
        
        for record in region.as_plain_texts():
            print(record)
        print("\n\n\n")
        faculty_data_harvester.find_names_in_region(region.as_plain_texts(), "video_test_comparing_models", "v2.2_gpt3.5turbo_illini_full", "txt")
        #if i == 22: break

if __name__ == "__main__":
    main()