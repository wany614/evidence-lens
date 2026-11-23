"""Sixteen opaque, deliberately different physical assemblies for the multiview study."""
from math import cos, sin, pi

from evidence_lens.blender_reference.formal_designs import Composition


def designs():
    rows=[]
    def new(n,slug,title,relation):
        c=Composition(n,slug,title,relation)
        rows.append(c)
        return c

    c=new(49,'cable_stayed_gantry','Fan-stayed loading gantry',
          'Two fan planes support an asymmetric deck. A raised rear ballast block is seen between near stays.')
    c.box('Long deck',(.04,-.02,.24),(1.22,.38,.085),'blue')
    for i,x in enumerate((-.52,.47)):
        c.box(f'Pedestal {i}',(x,.05,.10),(.15,.43,.20),'legs')
    for j,y in enumerate((-.17,.13)):
        c.rod(f'Leaning pylon {j}',(-.24,y,.24),(-.31,y,.83),.029,'terracotta')
        for k,x in enumerate((-.55,-.43,-.08,.13,.32,.55)):
            c.rod(f'Fan stay {j}-{k}',(-.31,y,.79-.017*(k%3)),(x,y,.277),.009,'cream')
    c.box('Rear ballast',(.34,.26,.185),(.30,.22,.37),'teal')
    c.box('Ballast cap',(.34,.26,.386),(.32,.23,.032),'chalk')

    c=new(50,'nested_paper_theatre','Offset paper theatre',
          'Three flat proscenia have progressively narrower, asymmetrically shifted windows. A crescent of the rear scene shows through both front masks.')
    c.box('Stage base',(0,.01,.055),(1.22,.73,.11),'legs')
    for i,(y,left,right,h,colour) in enumerate(((-.28,-.43,.37,.73,'rose'),(0,-.30,.36,.61,'blue'),(.22,-.27,.22,.52,'teal'))):
        c.box(f'Left paper wing {i}',((-.57+left)/2,y,.11+h/2),(left+.57,.045,h),colour)
        c.box(f'Right paper wing {i}',((right+.57)/2,y,.11+h/2),(.57-right,.045,h),colour)
        c.box(f'Header {i}',(0,y,.11+h),(1.14,.05,.075),colour)
    c.ball('Painted disc in back',(.12,.32,.46),(.215,.042,.21),'cream')
    c.cone('Low silhouette',(-.05,.12,.23),.18,0,.25,'terracotta')

    c=new(51,'spiral_ball_track','Stepped marble spiral',
          'A segmented descending spiral wraps a central column. Beads on upper and lower turns overlap across the source arc.')
    c.disk('Round plinth',(0,.03,.045),.46,.09,'dark')
    c.cone('Central pillar',(0,.03,.40),.075,.052,.68,'blue')
    spiral=[(.315*cos(i*pi/8),.03+.27*sin(i*pi/8),.72-i*.018) for i in range(31)]
    for j,(a,b) in enumerate(zip(spiral,spiral[1:])):
        c.rod(f'Track segment {j}',a,b,.026,'cream')
    for i in (3,13,25):
        x,y,z=spiral[i]
        c.ball(f'Marble {i}',(x,y,z+.047),(.043,.043,.043),'rose')

    c=new(52,'loom_beater','Table loom and suspended beater',
          'An angled front beater crosses coloured parallel warps seated between separate end beams.')
    for x in (-.46,.46):
        c.box(f'Loom side {x}',(x,.01,.16),(.085,.66,.13),'timber')
        c.rod(f'Beater arm {x}',(x,-.20,.19),(x,-.05,.56),.026,'terracotta')
    for y in (-.24,.27):
        c.rod(f'Yarn beam {y}',(-.53,y,.28),(.53,y,.28),.063,'dark')
    for i in range(14):
        x=-.38+i*.058
        c.rod(f'Warp {i}',(x,-.24,.325),(x,.27,.325),.008,('cream','teal','rose')[i%3])
    c.box('Swing reed',(0,-.06,.51),(1.02,.065,.095),'blue')
    c.ball('Boat shuttle',(.15,.04,.361),(.14,.045,.031),'cream')

    c=new(53,'staggered_roof_trusses','Three staggered roof trusses',
          'Triangular king-post frames repeat at distinct depths, with offset ridge height, longitudinal timbers and visible internal braces.')
    for i,(y,h,x) in enumerate(((-.30,.57,-.09),(.04,.79,.03),(.32,.65,.11))):
        left,right,peak=(-.53+x,y,.18),(.49+x,y,.18),(x,y,h)
        for name,a,b in [('tie',left,right),('rafter L',left,peak),('rafter R',right,peak),('king',(x,y,.18),peak)]:
            c.rod(f'{i} {name}',a,b,.035,'timber' if i!=1 else 'teal')
        for j,xx in enumerate((-.4,.4)):
            c.box(f'Pier {i} {j}',(xx+x,y,.08),(.13,.12,.16),'chalk')
    c.rod('Offset ridge',(-.09,-.30,.57),(.03,.04,.79),.025,'terracotta')
    c.rod('Rear ridge',(.03,.04,.79),(.11,.32,.65),.025,'terracotta')

    c=new(54,'domino_switchback','Domino switchback with a turning ramp',
          'A diagonal ramp separates three staggered banks of falling dominoes and hides the central connectors.')
    c.box('Foundation',(0,.02,.032),(1.23,.68,.064),'chalk')
    for j,(y,x0,dy) in enumerate(((-.24,-.46,.009),(.01,-.37,-.012),(.25,-.50,.006))):
        for i in range(6):
            c.box(f'Domino {j}-{i}',(x0+.17*i,y+dy*i,.21),(.044,.115,.30),('rose','dark','teal')[j],(0,7*i-13,0))
    c.box('Bridge turn',(.44,.13,.16),(.19,.37,.043),'cream',(19,0,0))

    c=new(55,'paired_hopper_chutes','Stepped hopper-chute line',
          'Two unequal wide mouths at opposite elevations narrow into offset chutes and partially cover a receiving tray.')
    for i,(x,y,z,r,m) in enumerate(((-.27,-.10,.52,.27,'blue'),(.30,.18,.62,.20,'terracotta'))):
        c.cone(f'Hopper {i}',(x,y,z),.065,r,.30,m)
        c.ring(f'Mouth rim {i}',(x,y,z+.15),r,.017,'cream',(0,0,0))
        c.rod(f'Chute {i}',(x,y,z-.13),(x+.09,y-.11,z-.35),.047,'dark')
        for sx in (-1,1):
            c.rod(f'Hopper leg {i} {sx}',(x+.16*sx,y,.02),(x+.11*sx,y,z),.018,'timber')
    c.box('Receiving tray',(.08,-.19,.075),(.62,.27,.09),'teal')

    c=new(56,'crossed_letterpress','Crossed lever letterpress',
          'Two oblique press arms meet above the platen. The upright rear rail and movable type bed sit at different depths.')
    c.box('Press bed',(0,-.05,.08),(.94,.63,.16),'dark')
    c.box('Type plate',(.04,-.19,.194),(.52,.27,.064),'chalk')
    for i,y in enumerate((-.05,.23)):
        c.rod(f'Pivot upright {i}',(-.33,y,.16),(-.33,y,.67),.053,'teal')
        c.rod(f'Rocker {i}',(-.34,y,.61),(.36,y,.76 if i==0 else .38),.036,'rose')
    c.rod('Handle tie',(.36,-.08,.76),(.36,.17,.76),.040,'cream')
    c.rod('Connecting link',(.04,-.05,.69),(.04,-.16,.35),.028,'cream')
    c.box('Moving platen',(.04,-.16,.32),(.58,.30,.065),'blue')

    c=new(57,'offset_bell_cage','Offset struck bell cage',
          'An asymmetric strut cage encloses three suspended stepped bells at different heights and a low striking ball.')
    for i,(x,y) in enumerate(((-.51,-.16),(.48,-.20),(-.33,.31),(.43,.27))):
        c.rod(f'Cage mast {i}',(x,y,.015),(x*.85,y,.80),.025,'dark')
    c.box('Top rack',(0,.05,.77),(1.0,.52,.055),'timber')
    for i,(x,y,h,r) in enumerate(((-.27,.11,.40,.17),(.05,-.15,.45,.11),(.29,.17,.54,.125))):
        c.rod(f'Hanger {i}',(x,y,.76),(x,y,h+.12),.011,'cream')
        c.cone(f'Bell body {i}',(x,y,h),r,.05,.22,('cream','rose','teal')[i])
        c.disk(f'Bell lip {i}',(x,y,h-.11),r+.015,.021,'dark')
    c.ball('Striker',(.01,.08,.17),(.073,.064,.08),'blue')

    c=new(58,'accordion_field_camera','Folding field camera',
          'Five faceted bellows connect an offset lens standard to a tall ground-glass back with a braced tripod.')
    for i,(x,y) in enumerate(((-.42,-.24),(.43,-.16),(.02,.35))):
        c.rod(f'Tripod {i}',(x,y,.015),(0,.02,.40),.033,'timber')
    c.box('Optical bed',(0,-.025,.43),(.65,.62,.065),'dark')
    for i in range(5):
        c.box(f'Pleated bellows {i}',(.025,-.15+i*.065,.61),(.39+.018*(i%2),.053,.29+.025*(i%2)),'teal')
    c.box('Glass back',(.025,.20,.65),(.54,.075,.47),'cream')
    c.box('Lens standard',(.025,-.19,.61),(.43,.069,.34),'rose')
    c.disk('Front lens barrel',(.025,-.27,.62),.102,.12,'dark',(90,0,0))

    c=new(59,'offset_vane_carousel','Offset vane carousel',
          'Radial slanted fins sweep above a two-tier platform. Several broad blades cover an eccentrically located rear chair.')
    c.disk('Turntable',(0,.01,.10),.45,.12,'blue')
    c.cone('Spindle',(0,.01,.41),.075,.05,.52,'dark')
    for i in range(7):
        a=2*pi*i/7
        c.box(f'Angled rotor vane {i}',(.24*cos(a),.01+.24*sin(a),.59),(.29,.055,.14),'cream' if i%2 else 'teal',(20,0,i*360/7))
    c.ball('Hub cap',(0,.01,.67),(.12,.12,.07),'rose')
    c.box('Eccentric seat',(.29,.13,.235),(.19,.16,.12),'rose')

    c=new(60,'tiered_sample_lathe','Miniature lathe with stepped stock',
          'A spindle and stepped chuck run along the x axis. The front tool carriage cuts across them in projection.')
    c.box('Lathe bed',(0,.02,.095),(1.24,.48,.19),'teal')
    for i,(x,w,h) in enumerate(((-.43,.23,.42),(.47,.17,.32))):
        c.box(f'Head {i}',(x,.13,.19+h/2),(w,.26,h),'blue')
    for i,(x,r,w,m) in enumerate(((-.27,.16,.12,'dark'),(-.13,.12,.16,'cream'),(.04,.079,.17,'timber'),(.21,.047,.17,'cream'))):
        c.disk(f'Coaxial stock {i}',(x,.12,.46),r,w,m,(0,90,0))
    c.box('Cross slide',(.03,-.14,.26),(.30,.23,.13),'rose')
    c.rod('Tool tip',(.04,-.24,.40),(.04,.098,.46),.026,'dark')

    c=new(61,'cascade_weir_model','Serpentine stepped weir',
          'Alternating dam walls bound an opaque stepped cascade. A high central partition hides low spillways.')
    for i,(y,z,dx) in enumerate(((-.27,.09,-.07),(-.02,.22,.04),(.24,.35,-.06))):
        c.box(f'Reservoir {i}',(dx,y,z/2),(1.04,.25,z),'teal')
        for j,x in enumerate((-.47,.47)):
            c.box(f'Bank {i}-{j}',(dx+x,y,z+.06),(.08,.26,.12),'chalk')
        c.box(f'Partial weir {i}',(dx+(-1)**i*.14,y-.12,z+.055),(.65,.045,.11),'rose')
    c.box('Central divider',(-.06,.04,.31),(.075,.54,.61),'cream')

    c=new(62,'semaphore_interlock','Mechanical semaphore interlock',
          'Two masts of different height carry opposing arms. Parallel drive links lie behind a foreground signal cabinet.')
    c.box('Interlock base',(0,.03,.055),(1.13,.53,.11),'dark')
    for i,(x,y,h,angle) in enumerate(((-.25,-.09,.79,29),(.29,.17,.61,-34))):
        c.rod(f'Mast {i}',(x,y,.11),(x,y,h),.031,'chalk')
        c.box(f'Semaphore arm {i}',(x+.10,y-.045,h-.04),(.38,.055,.069),'rose',(0,angle,0))
        c.disk(f'Signal hub {i}',(x,y-.065,h-.03),.055,.05,'dark',(90,0,0))
        c.rod(f'Drive link {i}',(x+.09,y+.05,.13),(x-.02,y+.05,h-.06),.012,'teal')
    c.box('Signal cabinet',(.08,-.18,.23),(.30,.18,.32),'blue')

    c=new(63,'offset_beam_balance','Three-pan torsion balance',
          'Unequal cantilevers on a single mast suspend three overlapping shallow pans by converging lines.')
    c.disk('Balance foundation',(0,.04,.042),.28,.084,'dark')
    c.rod('Mast',(0,.04,.08),(0,.04,.76),.035,'timber')
    for i,(x,y,r,h) in enumerate(((-.43,-.12,.14,.24),(.34,-.17,.16,.36),(.16,.28,.12,.32))):
        c.rod(f'Arm {i}',(0,.04,.71),(x,y,.69),.027,'teal')
        for j in range(3):
            a=j*2*pi/3
            c.rod(f'Line {i}-{j}',(x,y,.69),(x+r*cos(a),y+r*sin(a),h+.025),.009,'cream')
        c.cone(f'Pan {i}',(x,y,h),r*.80,r,.045,('blue','rose','cream')[i])

    c=new(64,'parabolic_facet_mount','Faceted acoustic reflector',
          'A concave mosaic sweeps backward from two offset uprights. A small front receiver masks different facets between the source endpoints.')
    c.box('Reflector shoe',(0,.10,.046),(.71,.44,.092),'dark')
    for x in (-.23,.23):
        c.rod(f'Tilt support {x}',(x,.15,.08),(x,.15,.49),.031,'timber')
    for j in range(3):
        for i in range(5):
            x=(i-2)*.19
            z=.28+j*.17
            y=.17-.37*(x*x+(z-.44)**2)
            c.box(f'Reflecting facet {j}-{i}',(x,y,z),(.17,.046,.156),'cream' if (i+j)%2 else 'chalk',((j-1)*7,0,(i-2)*6))
    c.rod('Receiver arm',(0,.13,.21),(.10,-.27,.48),.022,'teal')
    c.ball('Front receiver',(.10,-.27,.48),(.074,.068,.074),'rose')
    return rows
