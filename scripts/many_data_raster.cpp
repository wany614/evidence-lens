// Original CPU pinhole rasterizer for the six authored source families.
// Perspective-correct UV and camera-Z are interpolated independently of Gaussian predictions.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

struct Vertex { double x, y, z, u, v; };

static uint32_t mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    return x ^ (x >> 16);
}

static void color(double u, double v, int material, double shade, uint32_t seed, uint8_t* out) {
    static const double palette[][3] = {{167,131,103},{108,127,137},{164,102,72},{115,136,98},
                                        {157,151,128},{96,107,123},{169,146,110},{101,132,134}};
    double su = u * (4.1 + (material % 3) * .33), sv = v * (4.7 + (material % 4) * .27);
    int row = int(std::floor(sv));
    su += .36 * (row & 1);
    int col = int(std::floor(su));
    double a = su - col, b = sv - row;
    uint32_t bits = mix(seed ^ uint32_t(material * 5003) ^ (uint32_t(col)*2654435761U) ^ (uint32_t(row)*1597334677U));
    double value = .77 + (bits & 255) / 710.;
    if (std::min(std::min(a,1-a), std::min(b,1-b)) < .045) value *= .65;
    double px = .19 + ((bits >> 8) & 255) / 420.;
    double py = .19 + ((bits >> 16) & 255) / 420.;
    double dx = a - px, dy = b - py;
    if (dx*dx + dy*dy < .025 && (bits & 8)) value *= .52;
    if ((bits & 16) && std::abs(dx+((bits & 32) ? .55 : -.55)*dy) < .043 && std::abs(dy) < .25) value *= 1.35;
    value *= shade * (1.+.025*std::sin(41*u+13*v));
    const auto& base = palette[material % 8];
    for (int c=0;c<3;c++) out[c]=uint8_t(std::clamp(std::round(base[c]*value + int((bits>>(c*5))&7)-3.),0.,255.));
}

static Vertex near_edge(const Vertex& a, const Vertex& b) {
    const double t=(.05-a.z)/(b.z-a.z);
    return {a.x+t*(b.x-a.x),a.y+t*(b.y-a.y),.05,a.u+t*(b.u-a.u),a.v+t*(b.v-a.v)};
}

static void triangle(const Vertex* t, const double* k, int w, int h, int material,
                     double shade, uint32_t seed, float* zbuf, uint8_t* rgb) {
    double x[3], y[3], iz[3];
    for (int i=0;i<3;i++) {
        iz[i]=1./t[i].z; x[i]=k[0]*t[i].x*iz[i]+k[2]; y[i]=k[4]*t[i].y*iz[i]+k[5];
    }
    const double area=(y[1]-y[2])*(x[0]-x[2])+(x[2]-x[1])*(y[0]-y[2]);
    if (std::abs(area)<1e-9) return;
    const int x0=std::max(0,int(std::ceil(std::min({x[0],x[1],x[2]}))));
    const int y0=std::max(0,int(std::ceil(std::min({y[0],y[1],y[2]}))));
    const int x1=std::min(w-1,int(std::floor(std::max({x[0],x[1],x[2]}))));
    const int y1=std::min(h-1,int(std::floor(std::max({y[0],y[1],y[2]}))));
    for (int j=y0;j<=y1;j++) for (int i=x0;i<=x1;i++) {
        double l0=((y[1]-y[2])*(i-x[2])+(x[2]-x[1])*(j-y[2]))/area;
        double l1=((y[2]-y[0])*(i-x[2])+(x[0]-x[2])*(j-y[2]))/area;
        double l2=1-l0-l1;
        if (std::min({l0,l1,l2}) < -1e-9) continue;
        double a=l0*iz[0], b=l1*iz[1], c=l2*iz[2], z=1./(a+b+c);
        const int p=j*w+i;
        if (z>=zbuf[p]) continue;
        zbuf[p]=float(z);
        if (rgb) color(z*(a*t[0].u+b*t[1].u+c*t[2].u),z*(a*t[0].v+b*t[1].v+c*t[2].v),
                       material,shade,seed,rgb+3*p);
    }
}

extern "C" void paint(const double* triangles,const int32_t* materials,int count,
                      const double* k,const double* c2w,int w,int h,uint32_t seed,uint8_t* rgb,float* depth) {
    std::vector<float> zbuf(w*h,std::numeric_limits<float>::infinity());
    if (rgb) for (int j=0;j<h;j++) for(int i=0;i<w;i++) {
        uint8_t* p=rgb+3*(j*w+i);
        for(int c=0;c<3;c++) p[c]=uint8_t(125+13*c+12.*j/h);
    }
    for(int f=0;f<count;f++) {
        const double* face=triangles + 15*f;
        Vertex v[3], clipped[5];
        for(int i=0;i<3;i++) {
            const double* q=face+5*i;
            double x=q[0]-c2w[3], y=q[1]-c2w[7], z=q[2]-c2w[11];
            v[i] = {x*c2w[0]+y*c2w[4]+z*c2w[8], x*c2w[1]+y*c2w[5]+z*c2w[9],
                    x*c2w[2]+y*c2w[6]+z*c2w[10],q[3],q[4]};
        }
        int n=0;
        for(int i=0;i<3;i++) {
            const Vertex& a=v[(i+2)%3]; const Vertex& b=v[i];
            if ((a.z>=.05)!=(b.z>=.05)) clipped[n++]=near_edge(a,b);
            if(b.z>=.05) clipped[n++]=b;
        }
        if(n<3) continue;
        const double ux=face[5]-face[0],uy=face[6]-face[1],uz=face[7]-face[2];
        const double vx=face[10]-face[0],vy=face[11]-face[1],vz=face[12]-face[2];
        const double nx=uy*vz-uz*vy,ny=uz*vx-ux*vz,nz=ux*vy-uy*vx;
        const double light=(.27*nx-.49*ny+.83*nz)/std::sqrt(nx*nx+ny*ny+nz*nz);
        const double shade=.70+.31*std::max(0.,light);
        for(int i=1;i<n-1;i++) {
            const Vertex tri[3]={clipped[0],clipped[i],clipped[i+1]};
            triangle(tri,k,w,h,materials[f],shade,seed^mix(uint32_t(1+f/2)),zbuf.data(),rgb);
        }
    }
    if (depth) for (int i=0;i<w*h;i++) depth[i]=std::isfinite(zbuf[i]) ? zbuf[i] : std::numeric_limits<float>::quiet_NaN();
}
