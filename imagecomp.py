#!/usr/bin/env python
# encoding: utf-8
from __future__ import print_function
"""
imgcomp -- [re]compress an image multiple times, saving intermediate output

* https://en.wikipedia.org/wiki/JPEG
* https://en.wikipedia.org/wiki/Generation_loss
"""

from collections import namedtuple
import logging
import os
log = logging.getLogger()

from PIL import Image

ATTRS = ('n', 'filename', 'quality', 'md5', 'size')
ImageFile = namedtuple('ImageFile', ATTRS)

class _ImageFile(ImageFile):
    def __str__(self):
        return '%.2d %-32s %.3d %s %s' % self


import hashlib
def compute_image_metadata(filename, n, quality):
    """
    get filesize, get md5sum, pass through image info

    see also
    ---------
    * https://bitbucket.org/haypo/hachoir/src/default/hachoir-parser/hachoir_parser/image/jpeg.py
    * https://bitbucket.org/haypo/hachoir/src/default/hachoir-metadata/hachoir_metadata/jpeg.py?cl-106
    """
    stat = os.stat(filename)
    size = stat.st_size
    md5 = hashlib.new('md5')
    with open(filename,'rb') as f:
        md5.update(f.read())
    hashstr = md5.hexdigest()
    return _ImageFile(n, filename, quality, hashstr, size)


def imgcomp(src_image_path, times=20, quality=50):
    """
    Recompress an image with JPEG a number of times.
    """
    if isinstance(quality, (int, float, long)):
        qualityfunc = lambda n: quality
    else:
        qualityfunc = quality
    basename, ext = os.path.splitext(src_image_path)
    prev_img_path = src_image_path
    yield compute_image_metadata(src_image_path, 0, 0)
    try:
        for n in xrange(1, times+1):
            filename, ext = os.path.splitext(prev_img_path)
            new_image_path = "%s_%.2d.jpg" % (basename, n)
            _quality = qualityfunc(n)
            new_image = Image.open(prev_img_path)
            new_image.save(new_image_path, "JPEG", quality=_quality)
            prev_img_path = new_image_path
            yield compute_image_metadata(new_image_path, n, _quality)
    except IOError:
        log.error("Cannot convert %r" % src_image_path)
        raise


def iter_images(iterable, stop_after=0):
    stasis_count = 0
    prev = None
    for img in iterable:
        log.info(img)
        if prev is not None and (prev.md5 == img.md5):
            log.info("%.2d is the same as %.2d" % (img.n, prev.n))
            if stop_after:
                stasis_count += 1
                if stasis_count >= stop_after:
                    log.debug("stopping on %d after %d" %
                        (img.n, stasis_count))
                    #return img.n
                    break
        prev = img
    return img.n


def response_curve(image_filename, stop_after=1):
    def _response_curve(image_filename, stop_after=1):
        for quality in xrange(0,101):
            stasis_after = iter_images(
                    imgcomp(image_filename, times=200, quality=quality),
                    stop_after=stop_after) - 1
            yield(quality, stasis_after)
    try:
        log.setLevel(logging.ERROR)
        for n in _response_curve(image_filename, stop_after):
            print(n)
    finally:
        log.setLevel(logging.DEBUG) # TODO: [...]



import unittest
class Test_imgcomp(unittest.TestCase):
    def test_imgcomp_001_all(self):
        main('./tests/test_image.jpg', '--times=10')

    def test_imgcomp_002_stop_after_1(self):
        main('./tests/test_image.jpg', '--times=10', '--stop-after=1')

    def test_imgcomp_100_response_curve_after_1(self):
        main('./tests/test_image.jpg', '--response-curve')


def main(*args):
    import logging
    import optparse
    import sys

    prs = optparse.OptionParser(usage="%prog <src_image_path> <times>")

    prs.add_option('-T', '--times',
                   dest='times',
                   action='store',
                   type='int',
                   default=10,
                   help="times to [re]compress")

    prs.add_option('-Q', '--quality',
                   dest='quality',
                   action='store',
                   type='int',
                   default=50,
                   help="compression quality")

    prs.add_option('--quality-linear',
                   dest='quality_linear',
                   action='store_true',
                   help="increase quality linearly over steps")

    prs.add_option('-s', '--stop-after',
                   dest='stop_after',
                   action='store',
                   type='int',
                   default=0,
                   help="Stop after n changes (0 to keep going)")

    prs.add_option('-r', '--response-curve',
                   dest='response_curve',
                   action='store_true',
                   help='compute a quality response curve')


    prs.add_option('-v', '--verbose',
                    dest='verbose',
                    action='store_true',)
    prs.add_option('-q', '--quiet',
                    dest='quiet',
                    action='store_true',)
    prs.add_option('-t', '--test',
                    dest='run_tests',
                    action='store_true',)

    args = args and list(args) or sys.argv[1:]
    log.debug("main() args: %r", args)
    (opts, args) = prs.parse_args(args=args)

    if not opts.quiet:
        logging.basicConfig()

        if opts.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

    if opts.run_tests:
        sys.argv = [sys.argv[0]] + args
        import unittest
        sys.exit(unittest.main())

    if not len(args) == 1:
        print(args)
        prs.print_help()
        sys.exit(1)
    src_image_path = args[0]

    if opts.response_curve:
        return response_curve(src_image_path)

    quality = None
    if opts.quality_linear:
        quality = lambda n: int( (100.0 / opts.times) * n )
    elif opts.quality:
        quality = opts.quality

    images = imgcomp(src_image_path, times=opts.times, quality=quality)
    iter_images(images, stop_after=opts.stop_after)


if __name__ == "__main__":
    main()


