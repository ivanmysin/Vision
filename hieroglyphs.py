import matplotlib
matplotlib.rc('font', family='TakaoPGothic')
import matplotlib.pyplot as plt


hieroglyphs = '''
光は一般に、その固有波長が障害物よりも大きければ通過しやすい傾向にあり、この現象はレイリー散乱と呼ばれる。日中は長波長
の赤色光などは大気中を直線的に通過し、観察者の視野には光源である太陽の見た目の大きさの範囲に収まってしまう。一方短波長
の青色光は大気の熱的ゆらぎにより散乱するため空は青く見える。しかしながら夕方になると光線の入射角が浅くなり、
大気層を通過する距離が伸びる。すると青色光は障害物に衝突する頻度が増し、
かえって吸収されるなどの要因から地表に到達しにくくなる。代わって黄
、橙、赤などの長波長光線が散乱され、太陽が沈む方向の空が赤く見えることになる
年、世界中で鮮やかな夕焼けが確認された。これはクラカタウ火山の大噴火により大気中に障害物が撒き散らされたためである。
なお、火星においては大気による短波長の散乱よりちりによる長波長の散乱が卓越するため、ピンクの空と青い夕焼けが見られる。
'''
hieroglyphs = hieroglyphs.replace(" ", "")
hieroglyphs = hieroglyphs.replace("\n", "")

hieroglyphs = list( set(hieroglyphs) )


fig, axes = plt.subplots(figsize=(10, 10))

#axes.axis('off')
axes.set_xlim(0, 1)
axes.set_ylim(0, 1)
fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
axes.text(0.45, 0.45, hieroglyphs[0], fontsize=100)
axes.text(0.8, 0.45, hieroglyphs[1], fontsize=100)
fig.savefig("hieroglyphs.png", dpi=25)
plt.show()