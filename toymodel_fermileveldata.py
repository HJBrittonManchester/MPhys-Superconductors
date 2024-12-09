# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:13:46 2024

@author: zumai
"""

import numpy as np

# data for different fermi levels with the toy model
# also data for the dft at normal fermi level


r75 = np.array([[7.0, 0.0],
                [6.857142857142857, 0.0],
                [6.714285714285714, 0.0],
                [6.571428571428571, 0.0],
                [6.5, 0.0],
                [6.428571428571429, 7.28638916015625],
                [6.285714285714286, 12.672521972656252],
                [6.142857142857143, 16.42305603027344],
                [6.0, 19.52593688964844],
                [5.857142857142858, 22.248275756835938],
                [5.714285714285714, 24.73277587890625],
                [5.571428571428571, 27.00505065917969],
                [5.428571428571429, 29.156576538085936],
                [5.285714285714286, 31.20930786132812],
                [5.142857142857143, 33.15592651367187],
                [5.0, 35.0805908203125],
                [4.678571428571429, 39.11470184326172],
                [4.357142857142857, 42.98781433105469],
                [4.035714285714286, 46.752984619140626],
                [3.7142857142857144, 50.437655639648426],
                [3.392857142857143, 54.113179016113286],
                [3.071428571428571, 57.803338623046876],
                [2.75, 61.5996109008789],
                [2.4285714285714284, 65.5806655883789],
                [2.1071428571428568, 69.86450729370115],
                [1.7857142857142856, 74.3642333984375],
                [1.464285714285714, 78.99660034179686],
                [1.1428571428571423, 85.09624938964843],
                [0.8214285714285712, 92.7217254638672],
                [0.5, 96.04597930908204]])

r76 = np.array([[7.0, 0.0],
                [6.857142857142857, 0.0],
                [6.714285714285714, 0.0],
                [6.571428571428571, 0.0],
                [6.428571428571429, 0.0],
                [6.285714285714286, 5.83740234375],
                [6.142857142857143, 11.57114562988281],
                [6.0, 15.299725341796876],
                [5.857142857142858, 18.34406127929688],
                [5.714285714285714, 21.00419616699219],
                [5.571428571428571, 23.41185607910156],
                [5.428571428571429, 25.629244995117187],
                [5.285714285714286, 27.71124877929688],
                [5.142857142857143, 29.70909423828125],
                [5.0, 31.58619079589844],
                [5.0, 31.58619079589844],
                [4.678571428571429, 35.59468841552734],
                [4.357142857142857, 39.347052001953124],
                [4.035714285714286, 42.874383544921876],
                [3.7142857142857144, 46.25260848999024],
                [3.392857142857143, 49.47989730834961],
                [3.071428571428571, 52.59101104736328],
                [2.75, 55.628028869628906],
                [2.4285714285714284, 58.68883056640624],
                [2.1071428571428568, 61.90148315429687],
                [1.7857142857142856, 65.6986701965332],
                [1.464285714285714, 70.78658981323242],
                [1.1428571428571423, 77.78362274169922],
                [0.8214285714285712, 84.32967681884764],
                [0.5, 95.06580924987793]])

r74 = np.array([[7.0, 0.0],
                [6.857142857142857, 0.0],
                [6.714285714285714, 0.0],
                [6.571428571428571, 5.0616821289062495],
                [6.428571428571429, 11.99925537109375],
                [6.285714285714286, 16.273034667968748],
                [6.142857142857143, 19.7418212890625],
                [6.0, 22.691021728515622],
                [5.857142857142858, 25.37677001953125],
                [5.714285714285714, 27.828338623046882],
                [5.571428571428571, 30.151840209960938],
                [5.428571428571429, 32.35459289550781],
                [5.285714285714286, 34.454891967773435],
                [5.142857142857143, 36.50030517578125],
                [5.0, 38.45058288574219],
                [5.0, 38.45058288574219],
                [4.678571428571429, 42.69874877929688],
                [4.357142857142857, 46.76762084960938],
                [4.035714285714286, 50.71757354736328],
                [3.7142857142857144, 54.55409545898438],
                [3.392857142857143, 58.32841339111328],
                [3.071428571428571, 62.031379699707024],
                [2.75, 65.69226684570313],
                [2.4285714285714284, 69.3513244628906],
                [2.1071428571428568, 73.14027862548826],
                [1.7857142857142856, 77.4927276611328],
                [1.464285714285714, 83.7003189086914],
                [1.1428571428571423, 93.29619750976565],
                [0.8214285714285712, 104.24775695800783],
                [0.5, 109.26249542236329]])

r73 = np.array([[7.0, 0.0],
                [6.857142857142857, 0.0],
                [6.714285714285714, 4.900683593749999],
                [6.571428571428571, 12.29197998046875],
                [6.428571428571429, 16.719439697265628],
                [6.285714285714286, 20.0052734375],
                [6.142857142857143, 23.40087890625],
                [6.0, 26.013446044921878],
                [5.857142857142858, 28.53453674316406],
                [5.714285714285714, 30.887310791015622],
                [5.571428571428571, 33.11201782226562],
                [5.428571428571429, 34.992773437500006],
                [5.285714285714286, 37.272366333007824],
                [5.142857142857143, 39.25008697509765],
                [5.0, 41.17109222412109],
                [5.0, 41.17109222412109],
                [4.678571428571429, 45.360713195800784],
                [4.357142857142857, 49.42226715087891],
                [4.035714285714286, 53.42893524169922],
                [3.7142857142857144, 57.45755767822265],
                [3.392857142857143, 61.56850891113281],
                [3.071428571428571, 65.82948150634766],
                [2.75, 70.31731567382812],
                [2.4285714285714284, 75.05579528808593],
                [2.1071428571428568, 80.1601806640625],
                [1.7857142857142856, 85.88660583496095],
                [1.464285714285714, 94.6244354248047],
                [1.1428571428571423, 103.21773223876951],
                [0.8214285714285712, 108.55721206665038],
                [0.5, 117.49537506103516]])


r77 = np.array([[7.0, 0.0],
                [6.857142857142857, 0.0],
                [6.714285714285714, 0.0],
                [6.571428571428571, 0.0],
                [6.428571428571429, 0.0],
                [6.285714285714286, 0.0],
                [6.142857142857143, 0.0],
                [6.0, 8.77928466796875],
                [5.857142857142858, 12.588363647460938],
                [5.714285714285714, 15.541223144531251],
                [5.571428571428571, 18.03670043945312],
                [5.428571428571429, 20.265066528320315],
                [5.285714285714286, 22.306820678710938],
                [5.142857142857143, 24.10341796875],
                [5.0, 26.02442321777344],
                [5.0, 26.02442321777344],
                [4.678571428571429, 29.81520690917969],
                [4.357142857142857, 33.36632232666016],
                [4.035714285714286, 36.763757324218744],
                [3.7142857142857144, 40.13923797607421],
                [3.392857142857143, 43.522036743164065],
                [3.071428571428571, 46.97344284057617],
                [2.75, 50.54193878173828],
                [2.4285714285714284, 54.21929168701171],
                [2.1071428571428568, 57.96067810058594],
                [1.7857142857142856, 61.76609802246092],
                [1.464285714285714, 66.39114685058593],
                [1.1428571428571423, 80.43643951416016],
                [0.8214285714285712, 84.80535430908202],
                [0.5, 87.63060417175294]])

r78 = np.array([[7.0, 0.0],
                [6.857142857142857, 0.0],
                [6.714285714285714, 0.0],
                [6.571428571428571, 0.0],
                [6.428571428571429, 0.0],
                [6.285714285714286, 0.0],
                [6.142857142857143, 0.0],
                [6.0, 0.0],
                [5.857142857142858, 7.71083984375],
                [5.714285714285714, 11.64432678222656],
                [5.571428571428571, 14.534982299804689],
                [5.428571428571429, 16.993869018554687],
                [5.285714285714286, 19.189303588867187],
                [5.142857142857143, 21.19080810546875],
                [5.0, 23.089859008789066],
                [5.0, 23.089859008789066],
                [4.678571428571429, 27.006880187988287],
                [4.357142857142857, 30.640324401855473],
                [4.035714285714286, 34.120088195800776],
                [3.7142857142857144, 37.519352722167966],
                [3.392857142857143, 40.90215148925782],
                [3.071428571428571, 44.297756958007824],
                [2.75, 47.737271118164074],
                [2.4285714285714284, 51.24447784423828],
                [2.1071428571428568, 54.872433471679685],
                [1.7857142857142856, 58.67419433593749],
                [1.464285714285714, 62.733003997802726],
                [1.1428571428571423, 66.91988067626951],
                [0.8214285714285712, 70.80214080810546],
                [0.5, 74.19317245483398]])

r65 = np.array([[8.0, 0.0],
                [7.987060546875002, 0.0],
                [7.785714285714286, 17.62396240234375],
                [7.571428571428571, 25.495721435546876],
                [7.357142857142858, 31.599230957031256],
                [7.142857142857143, 36.794226074218756],
                [6.928571428571429, 41.470941162109376],
                [6.714285714285714, 45.769616699218744],
                [6.5, 49.80915222167968],
                [6.285714285714286, 53.64442443847656],
                [6.071428571428571, 57.2937255859375],
                [5.857142857142858, 60.86376037597657],
                [5.642857142857142, 64.31184692382814],
                [5.428571428571429, 67.67456970214845],
                [5.214285714285714, 70.97936706542968],
                [5.0, 74.23538513183594],
                [4.678571428571429, 78.93037109374998],
                [4.357142857142857, 83.8786163330078],
                [4.035714285714286, 88.74147338867186],
                [3.7142857142857144, 93.78578033447266],
                [3.392857142857143, 99.09052124023438],
                [3.071428571428571, 104.7218719482422],
                [2.75, 110.5090560913086],
                [2.4285714285714284, 116.1254638671875],
                [2.1071428571428568, 121.54761352539063],
                [1.7857142857142856, 126.80432357788087],
                [1.5, 131.8572998046875],
                [1.464285714285714, 132.5840362548828],
                [1.3, 136.7401123046875],
                [1.1, 149.615478515625],
                [0.8999999999999999, 158.57086181640625],
                [0.7, 165.28167724609375],
                [0.5, 170.72601318359375]])

r85 = np.array([[8.0, 0.0],
                [7.8, 0.0],
                [7.6, 0.0],
                [7.4, 0.0],
                [7.2, 0.0],
                [7.0, 0.0],
                [6.8, 0.0],
                [6.6, 0.0],
                [6.4, 0.0],
                [6.2, 0.0],
                [6.0, 0.0],
                [5.8, 0.0],
                [5.6, 0.0],
                [5.4, 0.0],
                [5.199999999999999, 0.0],
                [5.0, 0.0],
                [4.678571428571429, 4.589219665527343],
                [4.357142857142857, 6.934436798095702],
                [4.035714285714286, 8.642474365234373],
                [3.7142857142857144, 10.08298797607422],
                [3.392857142857143, 11.37487716674805],
                [3.071428571428571, 12.598170471191407],
                [2.75, 13.7825927734375],
                [2.4285714285714284, 15.051616668701172],
                [2.1071428571428568, 16.382376861572272],
                [1.7857142857142856, 17.838896179199217],
                [1.464285714285714, 19.458521270751955],
                [1.1428571428571423, 21.205810928344725],
                [0.8214285714285712, 23.01369361877442],
                [0.5, 25.39549522399902], ])

r72 = np.array([[7.0, 0.0],
                [7.0, 0.0],
                [7.0, 0.0],
                [7.0, 0.0],
                [7.0, 0.0],
                [7.0, 0.0],
                [7.0, 0.0],
                [6.888888888888889, 0.0],
                [6.777777777777778, 10.191894531250002],
                [6.666666666666667, 14.4877685546875],
                [6.555555555555555, 17.8387451171875],
                [6.444444444444445, 20.683178710937497],
                [6.333333333333333, 23.220764160156254],
                [6.222222222222222, 25.534301757812507],
                [6.111111111111111, 27.6773681640625],
                [6.0, 29.688928222656248],
                [6.535714285714286, 18.37734069824219],
                [6.071428571428571, 28.401913452148435],
                [5.607142857142858, 36.125274658203125],
                [5.142857142857142, 42.870941162109375],
                [4.678571428571429, 49.19180145263671],
                [4.214285714285714, 55.35682830810546],
                [3.75, 61.60083923339843],
                [3.2857142857142856, 68.14157409667968],
                [2.821428571428571, 75.26721801757813],
                [2.3571428571428568, 83.25741729736328],
                [1.8928571428571423, 91.80477447509764],
                [1.4285714285714288, 101.27859344482422],
                [0.9642857142857144, 112.29473648071291],
                [0.5, 128.2110939025879]])


fermi_levels_toy = np.array([[-0.85, 4.923828125],
                             [-0.84, 5.064453125],
                             [-0.83, 5.21923828125],
                             [-0.82, 5.369824218749999],
                             [-0.81, 5.5166259765625005],
                             [-0.8, 5.6259765625],
                             [-0.79, 5.8251953125],
                             [-0.78, 5.97705078125],
                             [-0.77, 6.137939453125],
                             [-0.76, 6.3369140625],
                             [-0.75, 6.5],
                             [-0.74, 6.6009765625],
                             [-0.73, 6.7426391601562505],
                             [-0.72, 6.886572265625],
                             [-0.71, 7.02021484375],
                             [-0.7, 7.1708984375],
                             [-0.69, 7.42236328125],
                             [-0.68, 7.579833984375],
                             [-0.67, 7.7490234375],
                             [-0.66, 7.9560546875],
                             [-0.65, 7.987060546875002],
                             [-0.6, 8.8118896484375],
                             [-0.59, 8.980224609375],
                             [-0.56, 9.0245361328125],
                             [-0.55, 8.05859375],
                             [-0.54, 7.1805419921875]])

r_DFT = np.array([[6.5, 0.0],
                  [6.071428571428571, 8.79490966796875],
                  [5.642857142857143, 12.48079528808594],
                  [5.214285714285714, 15.370968627929688],
                  [4.785714285714286, 17.91968688964844],
                  [4.357142857142858, 20.32740249633789],
                  [3.928571428571429, 22.712252807617187],
                  [3.5, 25.1832290649414],
                  [3.0714285714285716, 27.802067565917966],
                  [2.6428571428571432, 30.642699432373046],
                  [2.2142857142857144, 33.78362884521484],
                  [1.7857142857142856, 37.37805328369141],
                  [1.3571428571428577, 41.60355987548829],
                  [0.9285714285714288, 46.60267562866211],
                  [0.5, 54.244257354736334]])
