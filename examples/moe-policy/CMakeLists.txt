set(TARGET llama-moe-policy)
add_executable(${TARGET} moe-policy.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
