convert 'スクリーンショット 2026-01-28 092907.png' -crop 920x1860+400+0 1.png
convert 'スクリーンショット 2026-01-28 092921.png' -crop 920x1600+400+180 2.png
convert 'スクリーンショット 2026-01-28 092932.png' -crop 920x1600+400+180 3.png
convert 'スクリーンショット 2026-01-28 092946.png' -crop 920x1600+400+180 4.png
convert 'スクリーンショット 2026-01-28 092956.png' -crop 920x1908+400+180 5.png
~/bin/stitch_scroll.py 1.png 2.png 3.png 4.png 5.png
