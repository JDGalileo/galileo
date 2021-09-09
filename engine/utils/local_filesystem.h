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

#pragma once

#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>

#include "filesystem.h"
#include "utils/file_reader.h"
#include "utils/file_writer.h"
#include "utils/string_util.h"

#include "glog/logging.h"

namespace galileo {
namespace utils {

class LocalFileSystem : public FileSystem {
 public:
  LocalFileSystem() = default;
  ~LocalFileSystem() {}

 public:
  virtual bool Init(const FileConfig& file_config) { return true; }
  virtual std::shared_ptr<galileo::utils::IFileReader> OpenFileReader(
      const char* file_path) {
    LocalFileReader* local_file_reader = new LocalFileReader();
    if (!local_file_reader->Open(file_path)) {
      LOG(ERROR) << " Open file(" << file_path << ") failed";
      delete local_file_reader;
      return std::shared_ptr<galileo::utils::IFileReader>(nullptr);
    }
    return std::shared_ptr<galileo::utils::IFileReader>(local_file_reader);
  }
  virtual std::shared_ptr<galileo::utils::IFileWriter> OpenFileWriter(
      const char* file_path) {
    LocalFileWriter* local_file_writer = new LocalFileWriter();
    if (!local_file_writer->Open(file_path)) {
      LOG(ERROR) << " Open file(" << file_path << ") failed";
      delete local_file_writer;
      return std::shared_ptr<galileo::utils::IFileWriter>(nullptr);
    }
    return std::shared_ptr<galileo::utils::IFileWriter>(local_file_writer);
  }
  virtual bool IsFolderExist(const char* path) {
    if (NULL == path || path[0] == '\0') {
      LOG(ERROR) << " The path param is null";
      return false;
    }

    struct stat statbuf;
    if (!stat(path, &statbuf) == 0 && S_ISDIR(statbuf.st_mode)) {
      return false;
    }

    return true;
  }

  // check file is exist
  virtual bool IsFileExist(const char* path) {
    if (NULL == path || path[0] == '\0') {
      LOG(ERROR) << " The path param is null";
      return false;
    }

    struct stat statbuf;
    if (!stat(path, &statbuf) == 0 && S_ISREG(statbuf.st_mode)) {
      return false;
    }

    return true;
  }

  virtual bool ListFiles(const char* dir, std::vector<std::string>& files) {
    std::string targetDir(dir);
    if (!galileo::utils::endswith(dir, "/")) {
      targetDir.append("/");
      dir = targetDir.c_str();
    }

    files.clear();

    DIR* ptr_dir = opendir(dir);
    if (!ptr_dir) {
      LOG(ERROR) << " Open dir fail!"
                 << " dir:" << dir << " ,errno:" << errno;
      return false;
    }

    struct stat statbuf;
    struct dirent next_dir;

    struct dirent* ptr_next_dir = NULL;
    char sub_file[512] = {0};

    while (0 == readdir_r(ptr_dir, &next_dir, &ptr_next_dir)) {
      if (ptr_next_dir == NULL) {
        break;
      }

      sprintf(sub_file, "%s%s", dir, next_dir.d_name);
      if (stat(sub_file, &statbuf) == 0) {
        if (S_ISREG(statbuf.st_mode)) {
          files.push_back(sub_file);
        }
      }
    }
    closedir(ptr_dir);
    return true;
  }

  virtual bool ListFolders(const char* dir, std::vector<std::string>& floders) {
    std::string targetDir(dir);
    if (!galileo::utils::endswith(dir, "/")) {
      targetDir.append("/");
      dir = targetDir.c_str();
    }

    floders.clear();

    DIR* ptr_dir = opendir(dir);
    if (!ptr_dir) {
      LOG(ERROR) << " Open dir fail!"
                 << " dir:" << dir << " ,errno:" << errno;
      return false;
    }

    struct stat statbuf;
    struct dirent next_dir;

    struct dirent* ptr_next_dir = NULL;
    char sub_file[512] = {0};

    while (0 == readdir_r(ptr_dir, &next_dir, &ptr_next_dir)) {
      if (ptr_next_dir == NULL) {
        break;
      }

      sprintf(sub_file, "%s%s", dir, next_dir.d_name);
      if (stat(sub_file, &statbuf) == 0) {
        if (S_ISDIR(statbuf.st_mode)) {
          if (next_dir.d_name[0] == '.' && !next_dir.d_name[1]) {
          } else if (next_dir.d_name[0] == '.' && next_dir.d_name[1] == '.' &&
                     !next_dir.d_name[2]) {
          } else
            floders.push_back(sub_file);
        }
      }
    }
    closedir(ptr_dir);

    return true;
  }

  // rm file
  virtual bool RemoveFile(const char* path) {
    if (NULL == path || path[0] == '\0') {
      LOG(ERROR) << " The path param is null";
      return false;
    }

    if (unlikely(!IsFileExist(path))) {
      return true;
    }

    if (unlikely(unlink(path) != 0)) {
      LOG(ERROR) << " Remove file fail!"
                 << " file:" << path << " ,errno:" << errno;
      return false;
    }

    return true;
  }

  // rm dir
  virtual bool RemoveFolder(const char* path) {
    if (NULL == path || path[0] == '\0') {
      LOG(ERROR) << " The path param is null";
      return false;
    }

    if (unlikely(!IsFolderExist(path))) {
      return true;
    }

    std::vector<std::string> files;
    if (unlikely(!ListFiles(path, files))) {
      LOG(ERROR) << " Get the files of cur path failed!"
                 << " path:" << path;
      return false;
    }

    for (size_t pos = 0; pos < files.size(); ++pos) {
      if (unlikely(!RemoveFile(files[pos].c_str()))) {
        LOG(ERROR) << " Remove the file failed!"
                   << " file:" << files[pos];
        return false;
      }
    }
    files.clear();

    std::vector<std::string> folders;
    if (unlikely(!ListFolders(path, folders))) {
      LOG(ERROR) << " Get the folders of cur path failed!"
                 << " path:" << path;
      return false;
    }
    for (size_t pos = 0; pos < folders.size(); ++pos) {
      if (unlikely(!RemoveFolder(folders[pos].c_str()))) {
        LOG(ERROR) << " Remove the floder failed!"
                   << " floder:" << folders[pos];
        return false;
      }
    }
    folders.clear();
    if (0 != rmdir(path)) {
      LOG(ERROR) << " Remove the dir failed!"
                 << " dir:" << path << " ,errno:" << errno;
      return false;
    }

    return true;
  }

  // create one folder recursively
  virtual bool CreateDirRecursion(const char* dir) {
    if (NULL == dir || dir[0] == '\0') {
      LOG(ERROR) << " The path param is null";
      return false;
    }

    char tmp_dir[1024];
    strcpy(tmp_dir, dir);

    int len = strlen(tmp_dir);
    for (int idx = 1; idx <= len; ++idx) {
      if (tmp_dir[idx] != '/' && tmp_dir[idx] != '\0') {
        continue;
      }

      bool changed = false;
      if (tmp_dir[idx] == '/') {
        tmp_dir[idx] = '\0';
        changed = true;
      }

      if (access(tmp_dir, 0) == -1 && mkdir(tmp_dir, 0777) == -1) {
        LOG(ERROR) << " The dir created fail!"
                   << " dir:" << tmp_dir;
        return false;
      }

      if (changed) {
        tmp_dir[idx] = '/';
      }
    }

    return true;
  }
};

}  // namespace utils
}  // namespace galileo
