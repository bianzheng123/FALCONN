#ifndef __MATH_HELPERS_H__
#define __MATH_HELPERS_H__

namespace falconn {
    namespace core {

        inline int_fast64_t log2ceil(int_fast64_t x) {
            int_fast64_t res = 0;
            int_fast64_t cur = 1;
            while (cur < x) {
                cur *= 2;
                res += 1;
            }
            return res;
        }

        template<typename Point>
        struct NormalizationHelper {
            static void normalize(Point *) {
                static_assert(FalseStruct<Point>::value, "Point type not supported.");
            }

            template<typename T>
            struct FalseStruct : std::false_type {
            };
        };

        template<typename CoordinateType>
        struct NormalizationHelper<DenseVector<CoordinateType>> {
            static void normalize(DenseVector<CoordinateType> *p) { p->normalize(); }
        };

        template<typename Point>
        void normalize(Point *p) {
            NormalizationHelper<Point>::normalize(p);
        }

    }  // namespace core
}  // namespace falconn

#endif
