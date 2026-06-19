

# File template\_options.hpp



[**FileList**](files.md) **>** [**dynampi**](dir_f8db417ebd5c3d89eea80c24e3fd4369.md) **>** [**utilities**](dir_23e51883c93568b92bc8806003dcc116.md) **>** [**template\_options.hpp**](template__options_8hpp.md)

[Go to the source code of this file](template__options_8hpp_source.md)



* `#include <type_traits>`















## Classes

| Type | Name |
| ---: | :--- |
| struct | [**option\_value**](structoption__value.md) &lt;typename Option, Options&gt;<br> |
| struct | [**option\_value&lt; Option, Head, Tail... &gt;**](structoption__value_3_01Option_00_01Head_00_01Tail_8_8_8_01_4.md) &lt;typename Option, typename Head, Tail&gt;<br> |






















## Public Functions

| Type | Name |
| ---: | :--- |
|  consteval decltype(Option::value) | [**get\_option\_value**](#function-get_option_value) () <br> |




























## Public Functions Documentation




### function get\_option\_value 

```C++
template<typename Option, typename... Options>
consteval decltype(Option::value) get_option_value () 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/runner/work/DynaMPI/DynaMPI/include/dynampi/utilities/template_options.hpp`

