#include <cuda_runtime_api.h>

namespace adet {
int get_cudart_version() {
  return CUDART_VERSION;
}
} // namespace adet
