// Copyright 2020 JD.com, Inc. Galileo Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#include "value.h"

#include <rapidjson/prettywriter.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace quickjson {

// Value& m_value;
static value_type g_jsonNull;
static const char* parseErrorCodeName(rapidjson::ParseErrorCode code,
                                      char* buf) {
  switch (code) {
    case rapidjson::ParseErrorCode::kParseErrorNone:
      return "Success";
    case rapidjson::ParseErrorCode::kParseErrorDocumentEmpty:
      return "DocuemntEmpty";
    case rapidjson::ParseErrorCode::kParseErrorDocumentRootNotSingular:
      return "RootNotSingular";
    case rapidjson::ParseErrorCode::kParseErrorValueInvalid:
      return "ValueInvalid";
    case rapidjson::ParseErrorCode::kParseErrorObjectMissName:
      return "ObjectMissName";
    case rapidjson::ParseErrorCode::kParseErrorObjectMissColon:
      return "ObjectMissColon";
    case rapidjson::ParseErrorCode::kParseErrorObjectMissCommaOrCurlyBracket:
      return "ObjectMissCommaOrCurlyBracket";
    case rapidjson::ParseErrorCode::kParseErrorArrayMissCommaOrSquareBracket:
      return "ArrayMissCommaOrSquareBracket";
    case rapidjson::ParseErrorCode::kParseErrorStringUnicodeEscapeInvalidHex:
      return "StringUnicodeEscapseInvalidHex";
    case rapidjson::ParseErrorCode::kParseErrorStringUnicodeSurrogateInvalid:
      return "StringUnicodeSurrogateInvalid";
    case rapidjson::ParseErrorCode::kParseErrorStringEscapeInvalid:
      return "StringEscapeInvalid";
    case rapidjson::ParseErrorCode::kParseErrorStringMissQuotationMark:
      return "StringMissQuotationMark";
    case rapidjson::ParseErrorCode::kParseErrorStringInvalidEncoding:
      return "StringInvalidEncoding";
    case rapidjson::ParseErrorCode::kParseErrorNumberTooBig:
      return "NumberTooBig";
    case rapidjson::ParseErrorCode::kParseErrorNumberMissFraction:
      return "NumberMissFraction";
    case rapidjson::ParseErrorCode::kParseErrorNumberMissExponent:
      return "NumberMissExponent";
    case rapidjson::ParseErrorCode::kParseErrorTermination:
      return "Termination";
    case rapidjson::ParseErrorCode::kParseErrorUnspecificSyntaxError:
      return "UnspecificSyntaxError";
  }
  return "UnkonwnParseError";
}
static const char* sizeToString(size_t n, char* buf) {
  sprintf(buf, "%llu", (uint64_t)n);
  return buf;
}

Array::Array(Value& value)
    : m_jsonval(value.m_jsonval), m_allocator(value.m_allocator) {}

Array::Array(const Array& o)
    : m_jsonval(o.m_jsonval), m_allocator(o.m_allocator) {}
Array& Array::operator=(const Array& o) {
  m_jsonval = o.m_jsonval;
  m_allocator = o.m_allocator;
  return *this;
}

Array::~Array() {}

bool Array::empty() const { return m_jsonval->Empty(); }

size_t Array::size() const { return m_jsonval->Size(); }

void Array::reserve(size_t capacity) {
  m_jsonval->Reserve(capacity, *m_allocator);
}

Array& Array::append(const Value& value) {
  if (m_jsonval == value.m_jsonval) return *this;
  value_type jval;
  jval.CopyFrom(value.jsonValue(), *m_allocator);
  m_jsonval->PushBack(jval, *m_allocator);
  return *this;
}

Array& Array::append(bool value) {
  m_jsonval->PushBack(value, *m_allocator);
  return *this;
}

Array& Array::append(int32_t value) {
  m_jsonval->PushBack(value, *m_allocator);
  return *this;
}

Array& Array::append(uint32_t value) {
  m_jsonval->PushBack(value, *m_allocator);
  return *this;
}

Array& Array::append(int64_t value) {
  m_jsonval->PushBack(value, *m_allocator);
  return *this;
}

Array& Array::append(uint64_t value) {
  m_jsonval->PushBack(value, *m_allocator);
  return *this;
}

Array& Array::append(const char* value) {
  alloc_type& alloc = *m_allocator;
  m_jsonval->PushBack(g_jsonNull, alloc);
  value_type& elem = this->getElement(size() - 1);
  StringRefType strRef(value);
  elem.SetString(strRef, alloc);
  return *this;
}

Array& Array::append(const std::string& value) {
  alloc_type& alloc = *m_allocator;
  m_jsonval->PushBack(g_jsonNull, alloc);
  value_type& elem = this->getElement(size() - 1);
  StringRefType strRef(value.c_str(), value.size());
  elem.SetString(strRef, alloc);
  return *this;
}

Array& Array::append(double value) {
  m_jsonval->PushBack(value, *m_allocator);
  return *this;
}

Object Array::appendObject() {
  value_type value(ValueType::kObjectType);
  m_jsonval->PushBack(value, *m_allocator);
  value_type& elem = this->getElement(size() - 1);
  return Value(elem, *m_allocator).asObject();
}

Array& Array::append(const Array& value) {
  if (m_jsonval == value.m_jsonval) return *this;
  value_type jval;
  jval.CopyFrom(*value.m_jsonval, *m_allocator);
  m_jsonval->PushBack(jval, *m_allocator);
  return *this;
}

Array& Array::append(const Object& value) {
  if (m_jsonval == value.m_jsonval) return *this;
  value_type jval;
  jval.CopyFrom(*value.m_jsonval, *m_allocator);
  m_jsonval->PushBack(jval, *m_allocator);
  return *this;
}

Array Array::appendArray() {
  value_type value(ValueType::kArrayType);
  m_jsonval->PushBack(value, *m_allocator);
  value_type& elem = this->getElement(size() - 1);
  return Value(elem, *m_allocator).asArray();
}

Value Array::at(int index) {
  value_type& elem = this->getElement(index);
  return Value(elem, *m_allocator);
}

Value Array::at(int index) const {
  value_type& elem = this->getElement(index);
  return Value(elem, *m_allocator);
}

Value Array::operator[](int index) { return this->at(index); }

Value Array::operator[](int index) const { return this->at(index); }

value_type& Array::getElement(int index) const {
  if (index < 0 || index >= this->size()) return g_jsonNull;
  return (*m_jsonval)[index];
}

Array& Array::operator=(const Value& v) {
  m_jsonval = v.m_jsonval;
  m_allocator = v.m_allocator;
  return *this;
}

/// Object

Object::Object(Value& value)
    : m_jsonval(value.m_jsonval), m_allocator(value.m_allocator) {}

Object::Object(const Object& o)
    : m_jsonval(o.m_jsonval), m_allocator(o.m_allocator) {}

Object& Object::operator=(const Object& o) {
  m_jsonval = o.m_jsonval;
  m_allocator = o.m_allocator;
  return *this;
}

Object::~Object() {}

bool Object::empty() const { return m_jsonval->ObjectEmpty(); }

size_t Object::size() const { return m_jsonval->MemberCount(); }

// void Object::reserve(size_t capacity)
//{
//    m_jsonval->MemberReserve(capacity, *m_allocator);
//}

bool Object::has(const char* name) const {
  StringRefType sn(name);
  return m_jsonval->HasMember(sn);
}

bool Object::has(const std::string& name) const {
  StringRefType sn(name.c_str(), name.size());
  return m_jsonval->HasMember(sn);
}

Object& Object::set(const char* name, bool value) {
  StringRefType sn(name);
  value_type val(value);
  value_type& memval = this->_set(sn, val);
  return *this;
}

Object& Object::set(const char* name, int32_t value) {
  StringRefType sn(name);
  value_type val(value);
  value_type& memval = this->_set(sn, val);
  return *this;
}

Object& Object::set(const char* name, uint32_t value) {
  StringRefType sn(name);
  value_type val(value);
  value_type& memval = this->_set(sn, val);
  return *this;
}

Object& Object::set(const char* name, int64_t value) {
  StringRefType sn(name);
  value_type val(value);
  value_type& memval = this->_set(sn, val);
  return *this;
}

Object& Object::set(const char* name, uint64_t value) {
  StringRefType sn(name);
  value_type val(value);
  value_type& memval = this->_set(sn, val);
  return *this;
}

Object& Object::set(const char* name, double value) {
  StringRefType sn(name);
  value_type val(value);
  value_type& memval = this->_set(sn, val);
  return *this;
}

Object& Object::set(const char* name, const char* value) {
  value_type* keyp;
  value_type* valuep;
  StringRefType sn(name);
  StringRefType sv(value);
  alloc_type& alloc = *m_allocator;

  if (!this->_findMember(sn, &keyp, &valuep)) {
    m_jsonval->AddMember(sn, sv, alloc);
    this->_findMember(sn, &keyp, &valuep);
    keyp->SetString(sn, alloc);
    valuep->SetString(sv, alloc);
  } else {
    valuep->SetString(sv, alloc);
  }
  return *this;
}

Object& Object::set(const char* name, const std::string& value) {
  value_type* keyp;
  value_type* valuep;
  StringRefType sn(name);
  StringRefType sv(value.c_str(), value.size());
  alloc_type& alloc = *m_allocator;

  if (!this->_findMember(sn, &keyp, &valuep)) {
    m_jsonval->AddMember(sn, sv, alloc);
    this->_findMember(sn, &keyp, &valuep);
    keyp->SetString(sn, alloc);
    valuep->SetString(sv, alloc);
  } else {
    valuep->SetString(sv, alloc);
  }
  return *this;
}

Object& Object::set(const char* name, const Value& value) {
  value_type* keyp;
  value_type* valuep;
  StringRefType sn(name);
  alloc_type& alloc = *m_allocator;

  if (!this->_findMember(sn, &keyp, &valuep)) {
    m_jsonval->AddMember(sn, g_jsonNull, alloc);
    this->_findMember(sn, &keyp, &valuep);
    keyp->SetString(sn, alloc);
  }
  valuep->CopyFrom(value.jsonValue(), alloc);
  return *this;
}

Object& Object::set(const char* name, const Array& value) {
  value_type* keyp;
  value_type* valuep;
  StringRefType sn(name);
  alloc_type& alloc = *m_allocator;

  if (!this->_findMember(sn, &keyp, &valuep)) {
    m_jsonval->AddMember(sn, g_jsonNull, alloc);
    this->_findMember(sn, &keyp, &valuep);
    keyp->SetString(sn, alloc);
  }
  valuep->CopyFrom(*value.m_jsonval, alloc);
  return *this;
}

Object& Object::set(const char* name, const Object& value) {
  value_type* keyp;
  value_type* valuep;
  StringRefType sn(name);
  alloc_type& alloc = *m_allocator;

  if (!this->_findMember(sn, &keyp, &valuep)) {
    m_jsonval->AddMember(sn, g_jsonNull, alloc);
    this->_findMember(sn, &keyp, &valuep);
    keyp->SetString(sn, alloc);
  }
  valuep->CopyFrom(*value.m_jsonval, alloc);
  return *this;
}

Array Object::addArray(const char* name) {
  value_type* keyp;
  value_type* valuep;
  StringRefType sn(name);
  alloc_type& alloc = *m_allocator;
  if (!this->_findMember(sn, &keyp, &valuep)) {
    value_type sv(ValueType::kArrayType);
    m_jsonval->AddMember(sn, sv, alloc);
    this->_findMember(sn, &keyp, &valuep);
    keyp->SetString(sn, alloc);
  }
  return Value(*valuep, alloc).asArray();
}

Object Object::addObject(const char* name) {
  value_type* keyp;
  value_type* valuep;
  StringRefType sn(name);
  alloc_type& alloc = *m_allocator;
  if (!this->_findMember(sn, &keyp, &valuep)) {
    value_type sv(ValueType::kObjectType);
    m_jsonval->AddMember(sn, sv, alloc);
    this->_findMember(sn, &keyp, &valuep);
    keyp->SetString(sn, alloc);
  }
  return Value(*valuep, alloc).asObject();
}

Value Object::at(int index) const {
  if (index < 0 && index >= this->size()) return Value();
  value_type& val = (m_jsonval->MemberBegin().operator->() + index)->value;
  return Value(val, *m_allocator);
}

Value Object::getName(int index) const {
  if (index < 0 && index >= this->size()) return Value();
  value_type& name = (m_jsonval->MemberBegin().operator->() + index)->name;
  return Value(name, *m_allocator);
}

Value Object::operator[](const char* name) {
  StringRefType sn(name);
  value_type& memval = this->_set(sn, g_jsonNull, true);
  return Value(memval, *m_allocator);
}

Value Object::operator[](const std::string& name) {
  StringRefType sn(name.c_str(), name.size());
  value_type& memval = this->_set(sn, g_jsonNull, true);
  return Value(memval, *m_allocator);
}

bool Object::_findMember(StringRefType& name, value_type** keypp,
                         value_type** valuepp) {
  value_type& jvalue = *m_jsonval;
  miterator it = jvalue.FindMember(name);
  if (it == jvalue.MemberEnd()) return false;
  *keypp = &(*it).name;
  *valuepp = &(*it).value;
  return true;
}

Object& Object::operator=(const Value& v) {
  m_jsonval = v.m_jsonval;
  m_allocator = v.m_allocator;
  return *this;
}

value_type& Object::_set(StringRefType& sn, value_type& value,
                         bool if_not_exist) {
  value_type* keyp;
  value_type* valuep;
  alloc_type& alloc = *m_allocator;
  if (!this->_findMember(sn, &keyp, &valuep)) {
    m_jsonval->AddMember(sn, value, alloc);
    this->_findMember(sn, &keyp, &valuep);
    keyp->SetString(sn, alloc);
  } else {
    if (!if_not_exist) valuep->CopyFrom(value, alloc);
  }
  return *valuep;
}

//// Value

Value::Value()
    : m_allocator(&m_alloc),
      m_jsonval(&m_jsonValue),
      m_arr(*this),
      m_obj(*this) {}

Value::Value(alloc_type& alloc)
    : m_allocator(&alloc),
      m_jsonval(&m_jsonValue),
      m_arr(*this),
      m_obj(*this) {}

Value::~Value() {}

Value::Value(const Value& o)
    : m_allocator(o.m_allocator),
      m_jsonval(o.m_jsonval),
      m_arr(*this),
      m_obj(*this) {}

Value& Value::operator=(const Value& o) {
  m_allocator = o.m_allocator;
  m_jsonval->CopyFrom(*o.m_jsonval, *m_allocator);
  return *this;
}

bool Value::parse(const char* jsonStr, std::string* errMsg) {
  rapidjson::GenericDocument<encoding_type, alloc_type> doc(m_allocator);

  doc.Parse(jsonStr);
  if (!doc.HasParseError()) {
    m_jsonval->Swap(doc);
    return true;
  }
  if (errMsg) {
    char buf[256];
    rapidjson::ParseErrorCode code = doc.GetParseError();
    size_t errOffset = doc.GetErrorOffset();
    errMsg->append("Code:")
        .append(parseErrorCodeName(code, buf))
        .append(" Offset:")
        .append(sizeToString(errOffset, buf));
  }
  return false;
}

bool Value::parseFile(const char* filepath, std::string* errMsg) {
  FILE* fp = fopen(filepath, "rb");
  if (!fp) {
    if (errMsg) errMsg->assign("file not found");
    return false;
  }

  struct stat st;
  int fd = fileno(fp);
  if (fstat(fd, &st) != 0) {
    if (errMsg) {
      char msg[512];
      int code = errno;
      errMsg->assign(strerror_r(code, msg, sizeof(msg)));
    }
    fclose(fp);
    return false;
  }

  size_t fsize = st.st_size;
  char* buf = (char*)malloc(fsize + 1);

  size_t readed = fread(buf, 1, fsize, fp);
  if (readed != fsize) {
    if (errMsg) errMsg->assign("read file error");
    fclose(fp);
    free(buf);
    return false;
  }

  fclose(fp);
  buf[fsize] = '\0';

  const char* jsonstr = buf;
  bool ret = this->parse(jsonstr, errMsg);
  free(buf);
  return ret;
}

Value& Value::setNull() {
  jsonValue().~value_type();
  new ((void*)&jsonValue()) value_type();
  return *this;
}

Array& Value::toArray() {
  if (!this->isArray()) jsonValue() = value_type(ValueType::kArrayType);
  return m_arr;
}

Object& Value::toObject() {
  if (!this->isObject()) jsonValue() = value_type(ValueType::kObjectType);
  return m_obj;
}

void Value::clear() {
  this->setNull();
  if (m_allocator == &m_alloc) m_alloc.Clear();
}

Value& Value::operator=(bool value) {
  jsonValue().SetBool(value);
  return *this;
}

Value& Value::operator=(int32_t value) {
  jsonValue().SetInt(value);
  return *this;
}

Value& Value::operator=(uint32_t value) {
  jsonValue().SetUint(value);
  return *this;
}

Value& Value::operator=(int64_t value) {
  jsonValue().SetInt64(value);
  return *this;
}

Value& Value::operator=(uint64_t value) {
  jsonValue().SetUint64(value);
  return *this;
}

Value& Value::operator=(double value) {
  jsonValue().SetDouble(value);
  return *this;
}

Value& Value::operator=(const char* value) {
  StringRefType sref(value);
  jsonValue().SetString(sref, *m_allocator);
  return *this;
}

Value& Value::operator=(const std::string& value) {
  StringRefType sref(value.c_str(), value.size());
  jsonValue().SetString(sref, *m_allocator);
  return *this;
}

Value& Value::operator=(const Array& value) {
  if (value.m_jsonval == m_jsonval) return *this;
  m_jsonval->CopyFrom(*value.m_jsonval, *m_allocator);
  return *this;
}

Value& Value::operator=(const Object& value) {
  if (value.m_jsonval == m_jsonval) return *this;
  m_jsonval->CopyFrom(*value.m_jsonval, *m_allocator);
  return *this;
}

Value::operator bool() const { return jsonValue().GetBool(); }

Value::operator int16_t() const { return (int16_t)jsonValue().GetInt(); }

Value::operator uint16_t() const { return (uint16_t)jsonValue().GetUint(); }

Value::operator int32_t() const { return jsonValue().GetInt(); }

Value::operator uint32_t() const { return jsonValue().GetUint(); }

Value::operator int64_t() const { return jsonValue().GetInt64(); }

Value::operator uint64_t() const { return jsonValue().GetUint64(); }

Value::operator double() const { return jsonValue().GetDouble(); }

Value::operator const char*() const { return jsonValue().GetString(); }

const char* Value::getString(size_t* length) const {
  if (length) *length = jsonValue().GetStringLength();
  return jsonValue().GetString();
}

bool Value::isNull() const { return jsonValue().IsNull(); }

bool Value::isBool() const { return jsonValue().IsBool(); }

bool Value::isInt() const { return jsonValue().IsInt(); }

bool Value::isUInt() const { return jsonValue().IsUint(); }

bool Value::isIntegral() const {
  return jsonValue().IsInt() || jsonValue().IsUint() || jsonValue().IsInt64() ||
         jsonValue().IsUint64();
}

bool Value::isFloat() const { return jsonValue().IsDouble(); }
bool Value::isDouble() const { return jsonValue().IsDouble(); }

bool Value::isNumeric() const { return this->isIntegral() || this->isDouble(); }

bool Value::isString() const { return jsonValue().IsString(); }

bool Value::isArray() const { return jsonValue().IsArray(); }

bool Value::isObject() const { return jsonValue().IsObject(); }

Array& Value::asArray() { return this->toArray(); }

Object& Value::asObject() { return this->toObject(); }

std::string Value::asPrettyString() const {
  OutputStringStream os;
  rapidjson::PrettyWriter<string_buffer> writer(os);
  jsonValue().Accept(writer);
  return std::string(os.data(), os.size());
}

std::string Value::asString() const {
  size_t length;
  OutputStringStream os;
  const char* str = this->serialize(os, &length);
  return std::string(str, length);
}

const char* Value::serialize(OutputStringStream& os, size_t* length) const {
  os.clear();
  rapidjson::Writer<string_buffer> writer(os);
  jsonValue().Accept(writer);
  if (length) *length = os.size();
  return os.data();
}

size_t Value::size() const {
  if (this->isArray())
    return ((Value*)this)->asArray().size();
  else if (this->isObject())
    return ((Value*)this)->asObject().size();
  return 0;
}

Value Value::operator[](int index) { return (this->asArray())[index]; }

Value Value::operator[](const char* key) { return (this->asObject())[key]; }

Value Value::operator[](int index) const {
  return (((Value*)this)->asArray())[index];
}

Value Value::operator[](const char* key) const {
  return (((Value*)this)->asObject())[key];
}

}  // namespace quickjson
