"""Authored v1 static compositions, in metres above a shared studio floor.

The materialised geometry is frozen in configs/formal-scenes-v1.json. A composition
is a collection of named, opaque physical parts. Repetition within a mechanism
models its repeated parts. No family is sampled by changing a random seed.
"""
from math import cos, sin, pi


class Composition:
    def __init__(self, number, slug, title, relation):
        self.family = f"scene_{number:03d}_{slug}"
        self.title = title
        self.relation = relation
        self.objects = []

    def part(self, name, primitive, center, mat, **fields):
        self.objects.append({"id": len(self.objects)+10, "name": name, "primitive": primitive,
                             "center": [round(v, 6) for v in center], "material": mat, **fields})

    def box(self, name, at, size, mat, rot=None):
        self.part(name, "box", at, mat, size=size, **({"rotation_deg": rot} if rot else {}))

    def ball(self, name, at, radii, mat, rot=None):
        self.part(name, "ellipsoid", at, mat, radii=radii, **({"rotation_deg": rot} if rot else {}))

    def cone(self, name, at, r1, r2, height, mat, rot=None):
        self.part(name, "frustum", at, mat, bottom_radius=r1, top_radius=r2, depth=height,
                  **({"rotation_deg": rot} if rot else {}))

    def rod(self, name, a, b, radius, mat):
        delta = [round(y-x, 6) for x, y in zip(a, b)]
        center = [(x+y)/2 for x, y in zip(a, b)]
        self.part(name, "rod", center, mat, axis=delta, radius=radius)

    def ring(self, name, at, major, minor, mat, rot=(90, 0, 0)):
        self.part(name, "torus", at, mat, major_radius=major, minor_radius=minor, rotation_deg=rot)

    def disk(self, name, at, radius, thickness, mat, rot=None):
        self.cone(name, at, radius, radius, thickness, mat, rot)

    def wheel(self, name, x, y, z, radius, spokes):
        self.ring(name+" rim", (x,y,z), radius, .017, "dark")
        self.disk(name+" hub", (x,y,z), .035, .055, "cream", (90,0,0))
        for i in range(spokes):
            a = i*2*pi/spokes
            self.rod(f"{name} spoke {i+1}", (x,y,z), (x+radius*cos(a),y,z+radius*sin(a)), .005, "legs")


def authored():
    designs = []
    def new(*args):
        c = Composition(*args)
        designs.append(c)
        return c

    c = new(2,"book_nook","Two-bay book nook","A leaning oversized volume crosses the joint between open shelf bays and masks narrow rear spines.")
    for x in (-.61,.03,.61):
        c.box(f"Upright {x}",(x,.16,.42),(.042,.40,.84),"legs")
    for z in (.045,.43,.81):
        c.box(f"Shelf {z}",(0,.16,z),(1.25,.43,.045),"timber")
    for i,(x,y,z,w,h,m) in enumerate([(-.51,.13,.25,.07,.36,"blue"),(-.42,.10,.23,.08,.32,"cream"),
        (-.32,.16,.24,.09,.34,"green"),(-.20,.2,.23,.10,.32,"terracotta"),(.13,.17,.61,.08,.31,"green"),
        (.23,.17,.62,.09,.33,"cream"),(.37,.20,.58,.11,.24,"blue"),(.50,.16,.61,.08,.30,"terracotta")]):
        c.box(f"Bound volume {i+1}",(x,y,z),(w,.22,h),m)
    c.box("Leaning folio",(-.12,-.14,.32),(.16,.20,.63),"terracotta",(0,-19,0))
    c.box("Flat atlas",(.35,-.02,.12),(.38,.3,.11),"blue")

    c = new(3,"pulley_frame","Counterweighted pulley gantry","A front hanging bucket masks the rear cross-brace; unequal pulley wheels and cable runs sit at separate depths.")
    for x in (-.48,.48):
        c.box(f"Gantry foot {x}",(x,.12,.036),(.22,.46,.072),"dark")
        c.box(f"Gantry post {x}",(x,.18,.43),(.07,.09,.80),"blue")
    c.box("Gantry beam",(0,.16,.82),(1.13,.14,.09),"blue")
    c.rod("Back diagonal",(-.47,.29,.08),(.47,.29,.77),.021,"legs")
    for n,x,y,z,r in [("Main",-.22,-.10,.69,.115),("Return",.28,.11,.72,.078)]:
        c.wheel(n,x,y,z,r,4)
        c.rod(n+" axle",(x,y-.025,z),(x,.2,z),.024,"dark")
    c.rod("Suspension line",(-.33,-.10,.69),(-.33,-.10,.32),.010,"cream")
    c.rod("Counterweight line",(.36,.11,.72),(.36,.11,.35),.010,"cream")
    c.rod("Transfer line",(-.22,-.10,.80),(.28,.11,.80),.010,"cream")
    c.cone("Hanging bucket",(-.33,-.10,.23),.09,.14,.18,"terracotta")
    c.box("Counterweight",(.36,.11,.28),(.11,.13,.13),"green")

    c = new(4,"dish_drainer","Slotted dish drainer","Five inclined plates stand at increasing depths between side rails; their offset rims obscure one another.")
    c.box("Drip pan",(0,.04,.037),(1.05,.68,.074),"blue")
    for x in (-.46,.46):
        c.rod(f"Long rail {x}",(x,-.25,.18),(x,.33,.18),.019,"dark")
    for i,y in enumerate((-.23,-.11,.015,.14,.265)):
        x = (-.17,-.08,.015,.095,.165)[i]
        rr = (.265,.304,.246,.295,.257)[i]
        c.disk(f"Standing plate {i+1}",(x,y,.32),rr,.025,("cream","green","terracotta","cream","blue")[i],(75,-6,14))
        c.rod(f"Rack crossbar {i+1}",(-.46,y,.12),(.46,y,.12),.014,"dark")
    c.cone("Cutlery crock",(.40,-.19,.15),.065,.08,.18,"green")
    for i in range(3):
        c.rod(f"Spoon handle {i+1}",(.37+i*.032,-.19,.18),(.35+i*.040,-.18,.38+i*.025),.008,"legs")

    c = new(5,"stair_landing","Open stair and landing","Steps rise away from a side rail. Uprights and the open side of the top landing reveal distinct treads as the camera moves.")
    for i in range(5):
        y=-.39+i*.16; h=(i+1)*.105
        c.box(f"Solid step {i+1}",(-.09,y,h/2),(.72,.17,h),"timber")
        c.box(f"Contrasting nosing {i+1}",(-.09,y-.078,h-.011),(.73,.018,.022),"cream")
    c.box("Top landing",(.01,.42,.515),(.94,.24,.05),"green")
    for x,y,z in [(-.49,-.43,.095),(-.49,-.05,.31),(-.49,.31,.52),(.44,.42,0)]:
        c.rod(f"Support {x} {y}",(x,y,z),(x,y,z+.30 if z else .50),.018,"dark")
    c.rod("Slope handrail",(-.49,-.43,.395),(-.49,.31,.82),.025,"blue")
    c.box("Under-landing brace",(.38,.34,.25),(.08,.12,.5),"legs")

    c = new(6,"fruit_crate","Slatted fruit crate","Gaps between stacked front slats reveal irregular groups of round produce behind the same foreground fence.")
    c.box("Crate bed",(0,.04,.06),(1.08,.61,.11),"timber")
    for x in (-.50,.50):
        for y in (-.22,.30):
            c.box(f"Corner cleat {x} {y}",(x,y,.28),(.065,.063,.47),"legs")
    for j,z in enumerate((.14,.29,.44)):
        for y in (-.23,.30):
            c.box(f"Long slat {j+1} {y}",(0,y,z),(1.03,.027,.069),"timber")
        for x in (-.49,.49):
            c.box(f"End slat {j+1} {x}",(x,.035,z),(.03,.54,.07),"timber")
    for i,(x,y,z,rad,m) in enumerate([(-.32,-.02,.25,.13,"terracotta"),(-.10,.03,.26,.15,"green"),
        (.18,-.06,.27,.16,"cream"),(.36,.09,.24,.12,"terracotta"),(-.24,.14,.46,.145,"cream"),
        (.025,.10,.48,.15,"terracotta"),(.27,.13,.45,.13,"green")]):
        c.ball(f"Produce {i+1}",(x,y,z),(rad,rad*.89,rad),m)

    c = new(7,"surveyor_tripod","Surveyor's tripod","Three splayed legs converge on an instrument deck. The instrument and legs occlude a separate ruled target behind the open base.")
    c.box("Target board",(.18,.42,.43),(.56,.035,.78),"cream")
    for i in range(6):
        c.box(f"Target bar {i}",(.11,.394,.17+i*.096),(.35 if i%2==0 else .20,.012,.018),"blue")
    for i,(x,y) in enumerate([(-.47,-.23),(.36,-.27),(-.02,.37)]):
        c.rod(f"Tripod leg {i+1}",(x,y,.015),(-.07,.015,.61),.033,"legs")
    c.disk("Instrument deck",(-.07,.015,.64),.15,.055,"dark")
    c.box("Transit body",(-.06,.01,.76),(.23,.17,.17),"green")
    c.rod("Telescope",(-.28,-.04,.80),(.22,.05,.83),.055,"blue")
    c.ring("Optic collar",(.19,.045,.83),.047,.012,"cream",(0,86,10))

    c = new(8,"bench_vise","Bench vise and workpiece","A sliding jaw, a fixed jaw and a steel-colored screw surround a tall clamped workpiece; the cross handle passes in front.")
    c.box("Mounting slab",(0,.04,.07),(1.24,.65,.14),"timber")
    c.box("Vise base",(-.05,.09,.20),(.72,.36,.13),"blue")
    c.box("Fixed rear jaw",(-.09,.19,.39),(.46,.12,.29),"blue")
    c.box("Sliding jaw",(-.09,-.11,.36),(.46,.115,.28),"blue")
    c.box("Front jaw insert",(-.09,-.102,.495),(.46,.12,.05),"dark")
    c.box("Clamped timber",(-.10,.045,.50),(.18,.14,.54),"cream",(0,9,0))
    c.rod("Threaded screw",(-.10,-.35,.29),(-.10,.22,.29),.046,"dark")
    c.rod("Sliding T bar",(-.25,-.35,.12),(.07,-.35,.48),.015,"cream")
    c.disk("Loose washer",(.40,-.13,.16),.08,.027,"green")

    c = new(9,"pipe_manifold","Stepped pipe manifold","A pipe rises and crosses a rear valve bank before dropping into a lower return. Flange rims and handwheel overlap at different depths.")
    c.box("Plant foundation",(0,.03,.045),(1.15,.65,.09),"floor")
    c.box("Rear valve cabinet",(.16,.34,.36),(.68,.15,.57),"green")
    joints=[(-.41,-.17,.08),(-.41,-.17,.57),(.04,-.17,.57),(.04,.18,.57),(.37,.18,.57),(.37,.18,.10)]
    for i,(a,b) in enumerate(zip(joints,joints[1:])):
        c.rod(f"Pipe stretch {i+1}",a,b,.057,"terracotta")
    for i,p in enumerate(joints[1:-1]):
        c.ball(f"Elbow coupling {i+1}",p,(.075,.075,.075),"terracotta")
    for x in (-.32,-.12):
        c.disk(f"Flange {x}",(x,-.17,.57),.09,.036,"cream",(0,90,0))
    c.wheel("Valve handwheel",.22,-.13,.36,.107,5)
    c.rod("Valve stem",(.22,-.13,.36),(.22,.30,.36),.021,"dark")

    c = new(10,"open_drawers","Partly opened drawer stack","Three drawer fronts sit at unequal depths. An open middle drawer exposes an interior void and masks the lower drawer.")
    for x in (-.43,.43): c.box(f"Cabinet side {x}",(x,.12,.42),(.05,.51,.84),"green")
    c.box("Top slab",(0,.12,.84),(.92,.53,.045),"timber")
    c.box("Back panel",(0,.36,.41),(.83,.035,.80),"legs")
    for i,(z,y) in enumerate([(.14,-.075),(.395,-.35),(.66,-.125)]):
        c.box(f"Drawer {i+1} front",(0,y,z),(.82,.055,.21),"timber")
        c.box(f"Drawer {i+1} floor",(0,y+.18,z-.094),(.78,.38,.025),"cream")
        for x in (-.39,.39): c.box(f"Drawer {i+1} cheek {x}",(x,y+.18,z-.02),(.027,.38,.155),"legs")
        c.rod(f"Drawer {i+1} handle",(-.09,y-.055,z),(.09,y-.055,z),.022,"dark")
    c.ball("Knitted ball in open drawer",(.17,-.16,.40),(.078,.072,.08),"terracotta")

    c = new(11,"counting_frame","Counting frame with sliding beads","Four rods carry different bead groups in an open frame; a crosswise pointer sits in front of the lower rows.")
    for x in (-.51,.51): c.box(f"Frame stile {x}",(x,.08,.39),(.075,.14,.78),"legs")
    for z in (.047,.77): c.box(f"Frame rail {z}",(0,.08,z),(1.08,.14,.075),"timber")
    for j,xx in enumerate([[-.33,-.22,-.11,.32],[-.36,.08,.19,.30],[-.27,-.16,.19,.30],[-.31,.03,.14,.25]]):
        z=.18+j*.15
        c.rod(f"Counting rod {j+1}",(-.47,.07,z),(.47,.07,z),.013,"dark")
        for i,x in enumerate(xx):
            c.ball(f"Row {j+1} bead {i+1}",(x,.07,z),(.056,.077,.067),("terracotta","cream","blue","green")[j])
    c.rod("Pointer",(-.41,-.23,.04),(.18,-.04,.36),.018,"cream")

    c = new(12,"plant_stand","Tiered propagating stand","Plants at different heights break up the contour of two open shelves; sets of ellipsoidal leaves overlap pot rims and stems.")
    for x in (-.47,.43): c.box(f"Plant bracket {x}",(x,.20,.24),(.05,.25,.48),"legs")
    c.box("Upper shelf",(.02,.19,.47),(.98,.31,.055),"timber")
    c.box("Lower step",(-.18,-.14,.12),(.75,.35,.23),"green")
    for i,(x,y,z,r) in enumerate([(-.34,-.15,.23,.11),(.03,.18,.50,.13),(.35,.21,.50,.08)]):
        c.cone(f"Growing pot {i+1}",(x,y,z+.086),r*.67,r,.172,"terracotta")
        c.rod(f"Plant stem {i+1}",(x,y,z+.14),(x,y,z+.37),.012,"green")
        for j,(dx,dy) in enumerate([(-.064,-.035),(.07,.012),(.018,.07)]):
            c.ball(f"Plant {i+1} leaf {j+1}",(x+dx,y+dy,z+.26+j*.042),(.094,.038,.023),"green",(10,(-1)**j*24,j*45))

    c = new(13,"awning_kiosk","Striped awning kiosk","A sloping segmented roof overhangs a front counter. Recessed display shelves and stacked packages can be seen through the opening.")
    c.box("Kiosk counter",(0,-.01,.20),(1.03,.38,.40),"green")
    c.box("Counter lip",(0,-.04,.41),(1.11,.44,.044),"cream")
    c.box("Rear shelf unit",(0,.31,.46),(.96,.06,.69),"legs")
    for x in (-.49,.49): c.rod(f"Canopy prop {x}",(x,-.17,.43),(x,-.17,.81),.020,"dark")
    for i in range(7):
        x=-.47+i*.157
        c.box(f"Awning panel {i+1}",(x,.05,.84),(.158,.63,.045),"terracotta" if i%2==0 else "cream",(13,0,0))
    c.box("Recessed shelf",(0,.22,.60),(.85,.18,.031),"timber")
    for i,x in enumerate([-.28,.00,.23]):
        c.box(f"Stacked package {i+1}",(x,.16,.67),(.15,.12,.11),("blue","green","cream")[i])

    c = new(14,"slide_ladder","Ladder and playground slide","A high platform connects a sloping chute in front to an open ladder behind. The chute partially hides the platform supports.")
    c.box("Launch platform",(-.24,.17,.61),(.43,.34,.055),"blue")
    for x in (-.42,-.06):
        c.rod(f"Platform post {x}",(x,.24,0),(x,.24,.64),.025,"dark")
        c.rod(f"Rear ladder side {x}",(x,.52,.02),(x,.16,.64),.021,"green")
    for i in range(1,5):
        t=i/5; y=.52-.36*t; z=.02+.62*t
        c.rod(f"Ladder rung {i}",(-.42,y,z),(-.06,y,z),.014,"cream")
    c.box("Angled slide bed",(.10,-.14,.33),(.30,1.01,.058),"terracotta",(37,0,47))
    for x in (-.39,-.10): c.rod(f"Top handhold {x}",(x,.15,.64),(x,.15,.83),.022,"green")

    c = new(15,"jointed_lamp","Articulated drafting lamp","The lamp's folded arms and overhanging conical shade cross a tilted drawing board and its narrow front ledge.")
    c.disk("Weighted lamp base",(-.38,-.15,.044),.175,.088,"dark")
    path=[(-.38,-.15,.08),(-.49,-.02,.48),(-.13,-.09,.75),(.10,-.14,.68)]
    for i,(a,b) in enumerate(zip(path,path[1:])):
        c.rod(f"Lamp link {i+1}",a,b,.026,"blue")
        c.disk(f"Lamp joint {i+1}",b,.043,.06,"cream",(90,0,0))
    c.cone("Lamp shade",(.12,-.14,.61),.145,.049,.18,"green",(0,-17,0))
    c.box("Inclined drafting board",(.18,.22,.30),(.76,.045,.58),"timber",(-17,0,0))
    c.box("Drawing paper",(.17,.183,.34),(.57,.010,.38),"cream",(-17,0,0))
    c.box("Board ledge",(.18,.105,.052),(.78,.11,.033),"legs")

    c = new(16,"weaving_loom","Partly woven hand loom","A thick textile covers only the lower part of a taut warp; a diagonal shuttle crosses the otherwise open strands.")
    for x in (-.48,.48): c.box(f"Loom stile {x}",(x,.10,.43),(.06,.125,.84),"legs")
    for z in (.065,.81): c.rod(f"Warp roller {z}",(-.53,.1,z),(.53,.1,z),.047,"timber")
    for i in range(14):
        x=-.40+i*.061
        c.rod(f"Warp thread {i+1}",(x,.1,.1),(x,.1,.78),.0065,"cream")
    c.box("Woven section",(0,.076,.235),(.85,.040,.30),"blue")
    for j in range(3): c.box(f"Weft band {j}",(0,.052,.15+j*.085),(.84,.011,.018),"green")
    c.ball("Boat shuttle",(-.08,-.076,.435),(.245,.044,.033),"terracotta",(0,-17,0))

    c = new(17,"cycle_stand","Bicycle in a docking stand","Two spoked wheels, a triangular frame and an oblique front docking brace cross at three depths.")
    for n,x,y in [("Back",-.37,.10),("Front",.40,-.08)]: c.wheel(n,x,y,.27,.243,8)
    joints=[(-.37,.10,.27),(-.04,.03,.22),(-.16,.055,.61),(.23,-.035,.57),(.40,-.08,.27)]
    for j,(a,b) in enumerate([(0,1),(1,2),(2,0),(2,3),(3,1),(3,4)]):
        c.rod(f"Cycle frame tube {j+1}",joints[a],joints[b],.021,"green")
    c.ball("Saddle",(-.17,.05,.66),(.105,.052,.025),"dark")
    c.rod("Handlebar",(.23,-.12,.64),(.23,.06,.64),.018,"dark")
    for x in (.27,.53): c.rod(f"Dock leg {x}",(x,-.22,.015),(x,-.22,.39),.019,"blue")
    c.rod("Dock rail",(.27,-.22,.39),(.53,-.22,.39),.019,"blue")

    c = new(18,"balance_scale","Two-pan balance scale","Suspended pans form concavities on opposite sides of the pivot. The lower, nearer pan hides part of a separate weight stack.")
    c.disk("Scale base",(0,.10,.057),.22,.114,"dark")
    c.cone("Scale column",(0,.10,.34),.079,.039,.49,"green")
    c.rod("Balance beam",(-.40,-.035,.66),(.42,.20,.76),.022,"cream")
    for n,x,y,top in [("Left",-.40,-.035,.66),("Right",.42,.20,.76)]:
        for j,(dx,dy) in enumerate([(-.13,-.04),(.13,-.04),(0,.13)]):
            c.rod(f"{n} suspension {j+1}",(x,y,top),(x+dx,y+dy,top-.32),.008,"dark")
        c.cone(n+" pan",(x,y,top-.35),.076,.16,.06,"terracotta")
    for j,r in enumerate((.105,.076,.045)):
        c.disk(f"Stack weight {j+1}",(-.29,.22,.07+j*.09),r,.085,"blue")

    c = new(19,"corner_booth","Corner booth and pedestal table","Perpendicular seat backs meet at a corner. The offset table top and pedestal hide separate portions of the seat and a floor stool.")
    c.box("Rear bench seat",(.02,.31,.30),(.99,.28,.12),"green")
    c.box("Rear upholstered back",(.02,.45,.51),(.99,.09,.36),"blue")
    c.box("Return bench seat",(-.46,-.015,.30),(.25,.54,.12),"green")
    c.box("Return seat back",(-.58,-.015,.51),(.07,.57,.36),"blue")
    for x,y in [(-.41,.25),(.43,.25),(-.45,-.24)]:
        c.box(f"Seat support {x} {y}",(x,y,.14),(.09,.11,.28),"legs")
    c.disk("Cafe table top",(.10,-.13,.43),.30,.05,"timber")
    c.rod("Table pedestal",(.10,-.13,.04),(.10,-.13,.41),.055,"dark")
    c.disk("Foot plate",(.10,-.13,.033),.17,.05,"dark")
    c.cone("Extra stool",(.47,.07,.14),.105,.13,.28,"terracotta")

    c = new(20,"switchback_ramp","Switchback display ramp","Two solid inclined runs reverse direction at a landing; their railings and central barrier overlap across the return lane.")
    c.box("Lower eastbound run",(0,-.17,.14),(.96,.28,.055),"timber",(0,-14,0))
    c.box("Landing",(.46,.02,.26),(.27,.63,.06),"cream")
    c.box("Upper westbound run",(0,.19,.40),(.96,.28,.055),"timber",(0,16,0))
    for i,(x,y,z) in enumerate([(-.44,-.32,.04),(.0,-.32,.15),(.43,-.32,.26),(.43,.35,.28),(0,.35,.40),(-.43,.35,.52)]):
        c.rod(f"Guard post {i+1}",(x,y,z),(x,y,z+.23),.012,"dark")
    c.rod("Front railing",(-.44,-.32,.27),(.43,-.32,.49),.018,"blue")
    c.rod("Rear railing",(.43,.35,.51),(-.43,.35,.75),.018,"blue")
    c.box("Landing abutment",(.46,.02,.12),(.19,.57,.25),"green")
    c.box("High abutment",(-.44,.19,.25),(.16,.26,.50),"green")

    c = new(21,"drum_kit","Percussion cluster with cymbals","A large front-facing kick drum partly hides smaller raised drums; thin stands place tilted cymbals across the upper silhouettes.")
    c.disk("Kick shell",(0,-.05,.26),.25,.34,"terracotta",(90,0,0))
    c.disk("Kick head",(0,-.23,.26),.235,.015,"cream",(90,0,0))
    for i,(x,y,z,r) in enumerate([(-.30,-.12,.44,.145),(.27,.06,.51,.16),(-.12,.17,.58,.115)]):
        c.disk(f"Tom shell {i+1}",(x,y,z),r,.16,"blue",(-12,i*7,0))
        c.disk(f"Tom head {i+1}",(x,y-.016,z+.081),r*.96,.014,"cream",(-12,i*7,0))
        c.rod(f"Tom post {i+1}",(x,y,.03),(x,y,z-.08),.021,"dark")
    for i,(x,y,z) in enumerate([(-.48,.14,.71),(.43,-.09,.79)]):
        c.rod(f"Cymbal stand {i+1}",(x,y,.02),(x,y,z),.015,"dark")
        c.cone(f"Cymbal {i+1}",(x,y,z),.19,.038,.04,"cream",(8,(-1)**i*11,0))

    c = new(22,"picket_gate","Gate and planter","Pointed pickets and a diagonal gate brace mask a separate deep rectangular planter and its low foliage through narrow openings.")
    for x in (-.56,.56): c.box(f"Gate hinge post {x}",(x,-.15,.38),(.095,.095,.76),"legs")
    for i in range(8):
        x=-.45+i*.13; h=.53+.09*abs(x)/.45
        c.box(f"Picket {i+1}",(x,-.16,h/2),(.077,.044,h),"cream")
        c.part(f"Pointed cap {i+1}","frustum",(x,-.16,h+.047),"cream",bottom_radius=.055,top_radius=0.0,depth=.095,vertices=4,rotation_deg=[0,0,45])
    for z in (.15,.45): c.box(f"Rear rail {z}",(0,-.122,z),(1.05,.045,.061),"green")
    c.rod("Diagonal brace",(-.46,-.097,.1),(.47,-.097,.50),.029,"green")
    c.box("Planter trough",(.05,.30,.19),(.83,.31,.31),"terracotta")
    for j,x in enumerate([-.24,.00,.25]): c.ball(f"Shrub {j+1}",(x,.30,.41),(.16,.12,.14+j*.013),"green")

    c = new(23,"potters_wheel","Potter's wheel and tool rail","The stack of tapered vessel sections sits on a broad spindle disk. Upright tools in front partially obscure a second vase on a rear shelf.")
    c.box("Wheel base",(-.14,.00,.11),(.64,.49,.22),"green")
    c.rod("Wheel spindle",(-.14,0,.18),(-.14,0,.31),.066,"dark")
    c.disk("Turntable",(-.14,0,.32),.29,.035,"cream")
    for i,(z,rb,rt,h) in enumerate([(.41,.095,.155,.15),(.545,.155,.182,.12),(.65,.182,.07,.09),(.72,.07,.08,.05)]):
        c.cone(f"Thrown profile {i+1}",(-.14,0,z),rb,rt,h,"terracotta")
    c.ring("Vessel lip",(-.14,0,.751),.073,.011,"cream",(0,0,0))
    c.box("Rear pottery shelf",(.32,.33,.19),(.42,.24,.38),"timber")
    c.cone("Unfired vase",(.31,.32,.52),.078,.11,.26,"cream")
    c.box("Front tool block",(.33,-.22,.07),(.25,.17,.14),"legs")
    for i in range(4): c.rod(f"Modelling tool {i+1}",(.25+i*.045,-.23,.11),(.235+i*.055,-.20,.42+i*.033),.009,"dark")

    c = new(24,"orbital_mobile","Mechanical orbital mobile","A central sphere, tilted orbit hoops and satellites on crossed arms occlude one another inside a suspended frame.")
    c.box("Mobile foot",(-.43,.18,.045),(.37,.42,.09),"timber")
    c.rod("Upright cantilever",(-.47,.22,.07),(-.47,.22,.93),.025,"dark")
    c.rod("Top cantilever",(-.47,.22,.93),(.16,.07,.93),.022,"dark")
    c.rod("Suspension",(.16,.07,.93),(.16,.07,.59),.009,"cream")
    c.ball("Central globe",(.16,.07,.48),(.118,.118,.118),"terracotta")
    c.ring("Inclined orbit",(.16,.07,.48),.275,.013,"cream",(53,26,12))
    c.ring("Crossed orbit",(.16,.07,.48),.357,.011,"green",(89,-33,29))
    for i,(x,y,z,r) in enumerate([(-.092,-.04,.53,.060),(.445,.16,.62,.07),(.29,-.12,.26,.044),(-.16,.23,.38,.055)]):
        c.rod(f"Satellite arm {i+1}",(.16,.07,.48),(x,y,z),.009,"dark")
        c.ball(f"Satellite {i+1}",(x,y,z),(r,r,r),("blue","cream","green","blue")[i])
    return designs
