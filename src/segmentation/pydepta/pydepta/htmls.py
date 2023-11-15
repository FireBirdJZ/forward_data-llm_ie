from lxml import etree
from lxml.html.clean import Cleaner


class DomTreeBuilder(object):
    def __init__(self, html):
        if not isinstance(html, str) or not html.strip():
            raise ValueError("HTML content is empty or not provided.")
        
        cleaner = Cleaner(style=True, page_structure=False, remove_tags=('br',), safe_attrs_only=False)
        self.html = cleaner.clean_html(html)

    def build(self):
        parser = etree.HTMLParser(encoding='utf-8')
        return etree.fromstring(self.html, parser)