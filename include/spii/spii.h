#ifndef SPII_H
#define SPII_H

#ifdef _WIN32
#	ifdef spii_EXPORTS
#		define SPII_API __declspec(dllexport)
#		define SPII_API_EXTERN_TEMPLATE
#	else
#		define SPII_API __declspec(dllimport)
#		define SPII_API_EXTERN_TEMPLATE extern
#	endif
#else
#	define SPII_API
#	define SPII_API_EXTERN_TEMPLATE
#endif // WIN32

namespace spii
{

double wall_time();

void SPII_API check(bool expr, const char* message);
void SPII_API assertion_failed(const char* expr, const char* file, int line);

#define spii_assert(expr) if (!(expr)) { assertion_failed(#expr, __FILE__, __LINE__); }

#ifndef NDEBUG
	#define spii_dassert(expr) if (!(expr)) { assertion_failed(#expr, __FILE__, __LINE__); }
#else
	#define spii_dassert(expr) ((void)0)
#endif

}

#endif
