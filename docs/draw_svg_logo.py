import drawSvg as draw
from pathlib import Path
from embedded_svg_base64_font import EMBEDDED_FONT

class Hyperlink(draw.DrawingParentElement):
    TAG_NAME = 'a'
    def __init__(self, href, target=None, **kwargs):
        # Other init logic...
        # Keyword arguments to super().__init__() correspond to SVG node
        # arguments: stroke_width=5 -> stroke-width="5"
        super().__init__(href=href, target=target, **kwargs)


fpath = str(Path.cwd() / 'docs' / 'torchimize_logo.svg')

d = draw.Drawing(215, 80, origin=(0, 0), displayInline=False)

lettering = 'torchimize'
tlayer1 = draw.Text(lettering, 50, x=0, y=10, center=0.0, fill='#dddddd', font_family="FreightSansLFPro")
tlayer2 = draw.Text(lettering, 50, x=0, y=20, center=0.0, fill='#aaaaaa', font_family="FreightSansLFPro")
tlayer3 = draw.Text(lettering, 50, x=0, y=30, center=0.0, fill='#666666', font_family="FreightSansLFPro")
tlayer4 = draw.Text(lettering, 50, x=0, y=40, center=0.0, fill='#ee4c2c', font_family="FreightSansLFPro")

duration = '2s'
tlayer1.appendAnim(draw.Animate('y', duration, '-4;-16;-16;-4;', repeatCount='indefinite'))
tlayer2.appendAnim(draw.Animate('y', duration, '-12;-18;-18;-12;', repeatCount='indefinite'))
tlayer3.appendAnim(draw.Animate('y', duration, '-26;-20;-20;-26;', repeatCount='indefinite'))
tlayer4.appendAnim(draw.Animate('y', duration, '-34;-22;-22;-34', repeatCount='indefinite'))

# create hyperlink
hlink = Hyperlink('https://hahnec.github.io/torchimize/build/html/index.html', target='_blank', transform='skewY(0)')
hlink.append(tlayer1)
hlink.append(tlayer2)
hlink.append(tlayer3)
hlink.append(tlayer4)
d.append(hlink)

# save
d.saveSvg(fpath)
d.savePng(fpath.replace('.svg','.png'))

# customize svg with line replacement
to_be_replaced = '</defs>'
replaced_by = EMBEDDED_FONT+'</defs>'
if to_be_replaced:
    with open(fpath, 'r') as f, open(fpath.replace('.svg', '_font.svg'), 'w') as of:
        for line in f:
            if line.lower().startswith(to_be_replaced):
                of.write(replaced_by)
            else:
                of.write(line)
