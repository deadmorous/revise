/*
ReVisE: Remote visualization environment for large datasets
Copyright (C) 2021 Stepan Orlov, Alexey Kuzin, Alexey Zhuravlev, Vyacheslav Reshetnikov, Egor Usik, Vladislav Kiev, Andrey Pyatlin

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see https://www.gnu.org/licenses/agpl-3.0.en.html.

*/

// def_prop_class.h

/// \file
/// \brief Macros to declare property holder classes.

#pragma once

#include <boost/signals2.hpp>
#include <iostream>

namespace s3dmm {
namespace def_prop_class_util {

// See http://stackoverflow.com/questions/7943525/is-it-possible-to-figure-out-the-parameter-type-and-return-type-of-a-lambda

template< class T >
struct function_traits : public function_traits<decltype(&T::operator())> {};

template< class C, class R, class... Args >
struct function_traits<R(C::*)(Args...) const> {
    typedef R return_type;
    typedef std::tuple<Args...> args_type;
    static const size_t arity = sizeof...(Args);
    typedef std::function<R(Args...)> std_function_type;
};

struct prop_class_tag {};

} // end namespace def_prop_class_util
} // end namespace s3dmm

/// \brief Declares a class with a single private field and public getter and setter methods for it.
/// \param ClassName The name of the class.
/// \param PropType The type of the private field.
/// \param PassedPropType The type to use for setter parameter and const getter return value.
/// \param propGetter The name of the getter method.
/// \param propSetter The name of the setter method.
/// \note The field can be initialized by passing the initializer value to the constructor.
/// \note The class also has the const getter, named \a propGetter followed by \c constRef.
/// \note The typical use of this macro is as follows.
/// - Declare a class using this macro.
/// - Inherit the class by a class that needs the corresponding property.
/// \sa S3DMM_DECL_SIMPLE_CLASS_FIELD, S3DMM_DEF_NOTIFIED_PROP_CLASS, S3DMM_DEF_PROP_REF_CLASS.
#define S3DMM_DEF_PROP_CLASS(ClassName, PropType, PassedPropType, propGetter, propSetter) \
    class ClassName : public s3dmm::def_prop_class_util::prop_class_tag { \
    public: \
        explicit ClassName(PassedPropType value) : m_value(value) {} \
        ClassName() : m_value() {} \
        PassedPropType get() const { \
            return m_value; \
        } \
        void set(PassedPropType value) { \
            m_value = value; \
        } \
        PassedPropType propGetter() const { \
            return get(); \
        } \
        void propSetter(PassedPropType value) { \
            set(value); \
        } \
        PassedPropType propGetter##ConstRef() const { \
            return m_value; \
        } \
    private: \
        PropType m_value; \
    };


/// \brief Declares a class with a single private field, public getter and setter methods, and observers for value modification.
/// \param ClassName The name of the class.
/// \param PropType The type of the private field.
/// \param PassedPropType The type to use for setter parameter and,
/// const getter return value, and modification observer parameter.
/// \param propGetter The name of the getter method.
/// \param propSetter The name of the setter method.
/// \param addOnChangeObserver The name of the method to add modification observer.
/// \param removeOnChangeObserver The name of the method to remove modification observer.
/// \note The field can be initialized by passing the initializer value to the constructor.
/// \note The class also has the const getter, named \a propGetter followed by \c constRef.
/// \note Two kinds of observer callbacks are accepted: without arguments and with one argument of type \a PassedPropType.
/// If compiler fails to find the appropriate overload, one can use methods named \a addOnChangeObserver followed by
/// \c _0 (to add observers without arguments) or \a addOnChangeObserver followed by \c _1
/// (to add observers with one argument).
/// \note The typical use of this macro is as follows.
/// - Declare a class using this macro.
/// - Inherit the class by a class that needs the corresponding property.
/// .
///
/// Example:
/// \code
/// class Foo { ... };
/// S3DMM_DEF_NOTIFIED_PROP_CLASS(
///     WithFoo, Foo, const Foo&, foo, setFoo, onChangeFoo, offChangeFoo)
///
/// class Bar : public WithFoo {
/// public:
///     Bar() {
///         onChangeFoo([](){ cout << "foo has changed\n"; });
///     }
/// };
///
/// void baz() {
///     Bar bar;
///     bar.setFoo(Foo());    // Prints "foo has changed"
/// }
/// \endcode
/// \sa S3DMM_DECL_SIMPLE_CLASS_FIELD, S3DMM_DEF_PROP_CLASS, S3DMM_DEF_PROP_REF_CLASS.
#define S3DMM_DEF_NOTIFIED_PROP_CLASS(ClassName, PropType, PassedPropType, propGetter, propSetter, addOnChangeObserver, removeOnChangeObserver) \
    class ClassName : public s3dmm::def_prop_class_util::prop_class_tag { \
    public: \
        using value_type = PropType; \
        using reference_type = PassedPropType; \
        explicit ClassName(PassedPropType value) : m_value(value) {} \
        ClassName(const ClassName& value) : m_value(value.m_value) {} \
        ClassName& operator=(const ClassName& that) { \
            if (this != &that) \
                set(that.m_value); \
            return *this; \
        } \
        ClassName() : m_value() {} \
        PassedPropType get() const { \
            return m_value; \
        } \
        void set(PassedPropType value) { \
            m_value = value; \
            m_onChange( m_value ); \
        } \
        PassedPropType propGetter() const { \
            return get(); \
        } \
        void propSetter(PassedPropType value) { \
            set(value); \
        } \
        PassedPropType propGetter##ConstRef() const { \
            return get(); \
        } \
        boost::signals2::connection addOnChangeObserver##_0( const std::function<void()>& observer ) { \
            return addOnChangeObserverPriv( observer ); \
        } \
        boost::signals2::connection addOnChangeObserver##_1( const std::function<void(PassedPropType)>& observer ) { \
            return addOnChangeObserverPriv( observer ); \
        } \
        template< class F > \
        boost::signals2::connection addOnChangeObserver( F observer ) { \
            return addOnChangeObserverPriv( typename s3dmm::def_prop_class_util::function_traits<F>::std_function_type( observer ) ); \
        } \
        void removeOnChangeObserver( const boost::signals2::connection& observerId ) { \
            m_onChange.disconnect( observerId ); \
        } \
    private: \
        PropType m_value; \
        boost::signals2::signal< void(PassedPropType) > m_onChange; \
        boost::signals2::connection addOnChangeObserverPriv( const std::function<void()>& observer ) { \
            return m_onChange.connect( [observer](PassedPropType) { observer(); } ); \
        } \
        boost::signals2::connection addOnChangeObserverPriv( const std::function<void(PassedPropType)>& observer ) { \
            return m_onChange.connect( observer ); \
        } \
    };

/// \brief Declares class holding a reference to a value in a private field, with accessor methods.
/// \param ClassName The name of the class.
/// \param PropType The type of the private field (logically, some kind of reference, e.g., std::shared_ptr).
/// \param propRefAccessor The name of non-constant accessor method returning \a PropType&.
/// \param constPropRefAccessor The name of constant accessor method returning const \a PropType&.
/// \note The typical use of this macro is as follows.
/// - Declare a class using this macro.
/// - Inherit the class by a class that needs the corresponding property.
/// \sa S3DMM_DECL_SIMPLE_CLASS_FIELD, S3DMM_DEF_PROP_CLASS, S3DMM_DEF_NOTIFIED_PROP_CLASS.
#define S3DMM_DEF_PROP_REF_CLASS(ClassName, PropType, propRefAccessor, constPropRefAccessor) \
    class ClassName { \
    public: \
        ClassName() : m_value() {} \
        explicit ClassName(const PropType& value) : m_value(value) {} \
        PropType& propRefAccessor() { \
            return m_value; \
        } \
        const PropType& propRefAccessor() const { \
            return m_value; \
        } \
        const PropType& constPropRefAccessor() const { \
            return m_value; \
        } \
    private: \
        PropType m_value; \
    };

// Enable streaming
namespace std {

template<class T, enable_if_t<is_base_of<s3dmm::def_prop_class_util::prop_class_tag, T>::value, int> = 0>
ostream& operator<<(ostream& s, const T& x)
{
    s << x.get();
    return s;
}

template<class T, enable_if_t<is_base_of<s3dmm::def_prop_class_util::prop_class_tag, T>::value, int> = 0>
istream& operator>>(istream& s, T& x)
{
    typename T::value_type value;
    s >> value;
    x.set(value);
    return s;
}

}
