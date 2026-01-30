#if defined(__has_feature)
#  if __has_feature(thread_sanitizer)
#define TSAN_ENABLED
#  endif
#endif