

# File template\_options.hpp

[**File List**](files.md) **>** [**dynampi**](dir_f8db417ebd5c3d89eea80c24e3fd4369.md) **>** [**utilities**](dir_23e51883c93568b92bc8806003dcc116.md) **>** [**template\_options.hpp**](template__options_8hpp.md)

[Go to the documentation of this file](template__options_8hpp.md)


```C++
/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <type_traits>

template <typename Option, typename... Options>
struct option_value {
  static constexpr decltype(Option::value) value = Option::value;
};

template <typename Option, typename Head, typename... Tail>
struct option_value<Option, Head, Tail...> {
  static constexpr decltype(Option::value) value = [] {
    if constexpr (std::is_base_of_v<Option, Head>) {
      // Use Head’s value only if it really is derived from Option
      return static_cast<decltype(Option::value)>(Head::value);
    } else {
      return option_value<Option, Tail...>::value;
    }
  }();
};

template <typename Option, typename... Options>
consteval decltype(Option::value) get_option_value() {
  return option_value<Option, Options...>::value;
}
```


