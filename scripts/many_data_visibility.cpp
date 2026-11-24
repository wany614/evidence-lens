// Two-sided continuous ray/triangle intersections in the registered authored metric frame.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace {
using Vec = std::array<double,3>;
constexpr double near_z = 0.05;
constexpr double endpoint = 1e-6;
constexpr double edge = 1e-10;
constexpr double padding = 1e-10;

Vec sub(const Vec& a, const Vec& b) { return {a[0]-b[0],a[1]-b[1],a[2]-b[2]}; }
double dot(const Vec& a, const Vec& b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
Vec cross(const Vec& a, const Vec& b) {
    return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};
}
double length(const Vec& a) { return std::sqrt(dot(a,a)); }
Vec scale(const Vec& v, double t) { return {v[0]*t,v[1]*t,v[2]*t}; }

struct Triangle {
    Vec a,e1,e2,center,minimum,maximum;
    double normal_length;
    explicit Triangle(const double* v) {
        a={v[0],v[1],v[2]};
        Vec b{v[3],v[4],v[5]},c{v[6],v[7],v[8]};
        e1=sub(b,a); e2=sub(c,a);
        normal_length=length(cross(e1,e2));
        for (int k=0;k<3;++k) {
            center[k]=(a[k]+b[k]+c[k])/3.;
            minimum[k]=std::min({a[k],b[k],c[k]})-padding;
            maximum[k]=std::max({a[k],b[k],c[k]})+padding;
        }
    }
    double hit(const Vec& origin, const Vec& ray) const {
        auto p=cross(ray,e2);
        double det=dot(e1,p);
        if (std::abs(det)<=1e-12*normal_length) return -1;
        auto t=sub(origin,a);
        double u=dot(t,p)/det;
        if (u < -edge || u > 1+edge) return -1;
        auto q=cross(t,e1);
        double v=dot(ray,q)/det;
        if (v < -edge || u+v > 1+edge) return -1;
        return dot(e2,q)/det;
    }
};

struct Node {
    Vec lo{INFINITY,INFINITY,INFINITY},hi{-INFINITY,-INFINITY,-INFINITY};
    int first,last,left=-1,right=-1;
    bool overlaps(const Vec& origin, const Vec& direction,double t0,double t1) const {
        for (int k=0;k<3;++k) {
            if (direction[k] == 0) {
                if (origin[k]<lo[k] || origin[k]>hi[k]) return false;
            } else {
                double a=(lo[k]-origin[k])/direction[k], b=(hi[k]-origin[k])/direction[k];
                if (a>b) std::swap(a,b);
                t0=std::max(t0,a); t1=std::min(t1,b);
                if (t1<t0) return false;
            }
        }
        return true;
    }
};

class Scene {
    std::vector<Triangle> triangles;
    std::vector<int> order;
    std::vector<Node> nodes;
    int build(int first,int last) {
        Node node; node.first=first; node.last=last;
        for (int i=first;i<last;++i) {
            const auto& tri=triangles[order[i]];
            for (int k=0;k<3;++k) {
                node.lo[k]=std::min(node.lo[k],tri.minimum[k]);
                node.hi[k]=std::max(node.hi[k],tri.maximum[k]);
            }
        }
        int index=static_cast<int>(nodes.size());
        nodes.push_back(node);
        if (last-first>8) {
            auto extent=sub(node.hi,node.lo);
            int axis=static_cast<int>(std::max_element(extent.begin(),extent.end())-extent.begin());
            int mid=first+(last-first)/2;
            auto comp=[&](int a,int b){
                double x=triangles[a].center[axis],y=triangles[b].center[axis];
                return x==y ? a<b : x<y;
            };
            std::nth_element(order.begin()+first,order.begin()+mid,order.begin()+last,comp);
            int a=build(first,mid),b=build(mid,last);
            nodes[index].left=a; nodes[index].right=b;
        }
        return index;
    }
    bool trace(int id,const Vec& origin,const Vec& direction,double low,double& high,bool first) const {
        const Node& node=nodes[id];
        if (!node.overlaps(origin,direction,low,high)) return false;
        if (node.left>=0) {
            bool a=trace(node.left,origin,direction,low,high,first);
            if (a && first) return true;
            return trace(node.right,origin,direction,low,high,first) || a;
        }
        bool found=false;
        for (int i=node.first;i<node.last;++i) {
            double t=triangles[order[i]].hit(origin,direction);
            if (t>=low && t<=high) {
                high=t;
                if (first) return true;
                found=true;
            }
        }
        return found;
    }
public:
    Scene(const double* vertices,int n) : order(n) {
        triangles.reserve(n); nodes.reserve(2*n);
        for (int i=0;i<n;++i) triangles.emplace_back(vertices+9*i);
        std::iota(order.begin(),order.end(),0);
        build(0,n);
    }
    double nearest(const Vec& o,const Vec& d,double lo) const {
        double hi=std::numeric_limits<double>::infinity();
        return trace(0,o,d,lo,hi,false) ? hi : std::numeric_limits<double>::quiet_NaN();
    }
    bool occluded(const Vec& o,const Vec& d,double lo,double hi) const {
        return hi>=lo && trace(0,o,d,lo,hi,true);
    }
};

Vec origin(const double* c) { return {c[3],c[7],c[11]}; }
Vec to_world(const double* c,const Vec& d) {
    return {c[0]*d[0]+c[1]*d[1]+c[2]*d[2],c[4]*d[0]+c[5]*d[1]+c[6]*d[2],c[8]*d[0]+c[9]*d[1]+c[10]*d[2]};
}
Vec to_camera(const double* c,const Vec& d) {
    return {c[0]*d[0]+c[4]*d[1]+c[8]*d[2],c[1]*d[0]+c[5]*d[1]+c[9]*d[2],c[2]*d[0]+c[6]*d[1]+c[10]*d[2]};
}
} // namespace

extern "C" int visible_samples(const double* vertices,int ntri,const double* kt,const double* ct,
    const double* ks,const double* cs,int ns,const int32_t* xy,int np,
    double* points,double* optical,uint8_t* inside,uint8_t* visible) {
    if (ntri<=0 || np<=0 || ns<=0) return -1;
    Scene scene(vertices,ntri);
    auto center=origin(ct);
    for(int j=0;j<np;++j) {
        Vec camera_dir{(xy[2*j]-kt[2])/kt[0],(xy[2*j+1]-kt[5])/kt[4],1};
        double norm=length(camera_dir);
        auto ray=to_world(ct,scale(camera_dir,1/norm));
        double dist=scene.nearest(center,ray,near_z*norm);
        optical[j]=dist/norm;
        Vec point{};
        for (int k=0;k<3;++k) points[3*j+k]=point[k]=center[k]+ray[k]*dist;
        if (!std::isfinite(dist)) continue;
        for(int i=0;i<ns;++i) {
            const double* cam=cs+16*i;
            const double* k=ks+9*i;
            auto oc=origin(cam),direction=sub(point,oc);
            auto p=to_camera(cam,direction);
            if (p[2]<near_z) continue;
            double x=k[0]*p[0]/p[2]+k[2],y=k[4]*p[1]/p[2]+k[5];
            if (x<-.5 || x>=377.5 || y<-.5 || y>=251.5) continue;
            int index=i*np+j;
            inside[index]=1;
            double d=length(direction);
            visible[index]=!scene.occluded(oc,scale(direction,1/d),near_z*d/p[2],d-endpoint);
        }
    }
    return 0;
}
