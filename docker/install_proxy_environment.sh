#!/bin/sh
# As a former Huawei contractors, we provide helper scripts to make life of
# current Huawei employees easier. This file sets various services to pass
# through the firewall specified by `https_proxy`. In fact, it doesn't do any
# Huawei-specific settings explicitly, and it should not affect systems with
# direct Internet access.

. /etc/profile

if test -z "$https_proxy" ; then
  echo "https_proxy is not set. Probably, you don't need a proxy. Skipping proxy-environment setup" >&2
  exit 0
fi

cacerts=/usr/local/share/ca-certificates/Custom.crt

if test -d "$JAVA_HOME" && test -f "$cacerts" ; then
  JAVA_KEYSTORE=""
  for keystore in $JAVA_HOME/lib/security/cacerts \
                  $JAVA_HOME/jre/lib/security/cacerts ; do
    echo "Checking Java keystore $keystore"
    if test -f "$keystore" ; then
      JAVA_KEYSTORE="$keystore"
      break
    fi
  done

  if test -n "$JAVA_KEYSTORE" ; then
    echo "export JAVA_KEYSTORE=\"$JAVA_KEYSTORE\"" >> /etc/profile
  else
    echo "Java keystore wasn't found" >&2
    exit 1
  fi

  pems_dir=/tmp/pems
  rm -rf "$pems_dir" 2>/dev/null || true
  mkdir "$pems_dir"
  (
  cd "$pems_dir"
  awk 'BEGIN {c=0;doPrint=0;}
       /END CERT/ {print > "cert." c ".pem";doPrint=0;}
       /BEGIN CERT/{c++;doPrint=1;} { if(doPrint == 1) {print > "cert." c ".pem"} }' < $cacerts
  for f in `ls cert.*.pem`; do
    keytool -import -trustcacerts -noprompt \
            -keystore "$JAVA_KEYSTORE" -alias "`basename $f`" \
            -file "$f" -storepass changeit;
    if test "$?" != "0" ; then
      echo "Failed to install sertificate $f to $JAVA_KEYSTORE" >&2
      exit 1
    fi
  done
  )
  rm -rf "$pems_dir"
else
  {
  echo "JAVA_HOME is not set or '$cacerts' file is not installed."
  echo "Did you run \`install_huawei_certificates.sh\` before running this script?"
  echo "Skipping Java certificates setup."
  } >&2
fi

PROXY_HOST=`echo $https_proxy | sed 's@.*//\(.*\):.*@\1@'`
PROXY_PORT=`echo $https_proxy | sed 's@.*//.*:\(.*\)@\1@'`
{
echo "export http_proxy=$http_proxy"
echo "export https_proxy=$https_proxy"
echo "export HTTP_PROXY=$http_proxy"
echo "export HTTPS_PROXY=$https_proxy"
echo "export GRADLE_OPTS='-Dorg.gradle.daemon=false -Dandroid.builder.sdkDownload=true -Dorg.gradle.jvmargs=-Xmx2048M -Dhttp.proxyHost=$PROXY_HOST -Dhttp.proxyPort=$PROXY_PORT -Dhttps.proxyHost=$PROXY_HOST -Dhttps.proxyPort=$PROXY_PORT'"
echo "export MAVEN_OPTS='-Dhttps.proxyHost=$PROXY_HOST -Dhttps.proxyPort=$PROXY_PORT -Dmaven.wagon.http.ssl.insecure=true'"
echo "export no_proxy=localhost,127.0.0.0,127.0.1.1,127.0.1.1,.huawei.com"
echo "export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt"
echo "export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt"
echo "export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt"
} >>/etc/profile

if test -f /etc/sudoers ; then
{
echo "Defaults        env_keep += \"http_proxy HTTP_PROXY https_proxy HTTPS_PROXY no_proxy GRADLE_OPTS MAVEN_OPTS\""
} >>/etc/sudoers
else
  echo "/etc/sudoers is not updated" >&2
fi

if test -f /etc/wgetrc; then
  echo ca_certificate=/etc/ssl/certs/ca-certificates.crt >> /etc/wgetrc
else
  echo "/etc/wgetrc is not updated" >&2
fi

mkdir /root/.android/
cat >/root/.android/androidtool.cfg <<EOF
http.proxyHost=$PROXY_HOST
http.proxyPort=$PROXY_PORT
https.proxyHost=$PROXY_HOST
https.proxyPort=$PROXY_PORT
EOF


if test -f "$JAVA_KEYSTORE"; then
echo "Creating /etc/bazel.bazelrc"
cat >/etc/bazel.bazelrc <<EOF
startup --host_jvm_args=-Djavax.net.ssl.trustStore=$JAVA_KEYSTORE \
        --host_jvm_args=-Djavax.net.ssl.trustStorePassword=changeit
EOF
else
  echo "No Java keystore found. Not creating /etc/bazel.bazelrc" >&2
fi


# TODO: Handle Pythonish import ssl; ssl._create_default_https_context = ssl._create_unverified_context;
# TODO: Handle Pythonish certifi internal certificates
if test -x `which pip` ; then
  pip config --global set global.cert /etc/ssl/certs/ca-certificates.crt
else
  echo "pip not found, SSL cert not updated. Install pip before executing $0" >&2
fi

if test -x `which pip3` ; then
  pip3 config --global set global.cert /etc/ssl/certs/ca-certificates.crt
else
  echo "pip3 not found, SSL cert not updated. Install pip3 before executing $0" >&2
fi

